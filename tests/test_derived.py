"""Behavior tests for derived erasure targets: residualized descriptors, rank-matched category."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ResidualizedDescriptorTest(unittest.TestCase):
    def test_descriptor_is_residualized_on_train_category_means(self):
        from eegdistill.derived import add_derived_labels

        train = {"edges": np.array([1.0, 3.0, 10.0, 14.0]),
                 "cat": np.array(["A", "A", "B", "B"])}
        test = {"edges": np.array([5.0, 9.0]), "cat": np.array(["A", "B"])}
        z = np.zeros((4, 3))
        new_train, new_test, derived = add_derived_labels(
            z, train, test, category="cat", residualize=["edges"], rank_matched=False)
        # Train means: A = 2, B = 12. Test images use the train means of their category.
        np.testing.assert_allclose(new_train["edges_resid_cat"], [-1, 1, -2, 2])
        np.testing.assert_allclose(new_test["edges_resid_cat"], [3, -3])
        self.assertEqual(derived["edges_resid_cat"],
                         {"kind": "residualized", "from": "edges", "on": "cat"})
        np.testing.assert_array_equal(new_train["edges"], train["edges"])  # originals kept


def four_category_embeddings():
    """Strong A-vs-B axis (dim 0), weak C-vs-D axis (dim 1), category-free dim 2."""
    code = {"A": (1.0, 0.0), "B": (-1.0, 0.0), "C": (0.0, 0.1), "D": (0.0, -0.1)}
    cats = np.repeat(list("ABCD"), 4)
    z = np.array([[*code[c], (-1.0) ** i] for i, c in enumerate(cats)])
    test_cats = np.array(list("ABCD"))
    test_z = np.array([[*code[c], 0.5] for c in test_cats])
    return z, cats, test_z, test_cats


class RankMatchedCategoryTest(unittest.TestCase):
    def test_rank_matched_label_follows_only_the_top_category_axis(self):
        from eegdistill.derived import add_derived_labels

        z, cats, _, test_cats = four_category_embeddings()
        new_train, new_test, derived = add_derived_labels(
            z, {"cat": cats}, {"cat": test_cats}, category="cat")
        label = new_train["cat_rank1"]
        self.assertAlmostEqual(abs(np.corrcoef(label, z[:, 0])[0, 1]), 1.0)
        np.testing.assert_allclose(label[cats == "C"], 0, atol=1e-12)
        np.testing.assert_allclose(label[cats == "D"], 0, atol=1e-12)
        self.assertAlmostEqual(new_test["cat_rank1"][0], label[0])  # same category, same value
        self.assertEqual(derived["cat_rank1"], {"kind": "rank_matched", "from": "cat", "rank": 1})

    def test_erasing_rank_matched_label_is_rank_one_and_leaves_weaker_category_axis(self):
        from eegdistill.derived import add_derived_labels
        from eegdistill.erasure import evaluate_erasure

        z, cats, test_z, test_cats = four_category_embeddings()
        train, test, _ = add_derived_labels(z, {"cat": cats}, {"cat": test_cats}, category="cat")
        report = evaluate_erasure(z, test_z, test_z, train, test, categorical=("cat",), controls=2)
        rank1 = report["interventions"]["cat_rank1"]
        self.assertEqual(rank1["rank"], 1)
        self.assertEqual(report["interventions"]["cat"]["rank"], 2)
        self.assertEqual(report["baseline"]["probes"]["cat"]["balanced_accuracy"], 1.0)
        # A and B collapse together; C and D stay separable on the weak axis.
        self.assertEqual(rank1["target"]["probes"]["cat"]["balanced_accuracy"], 0.75)


class RunnerTest(unittest.TestCase):
    def test_runner_adds_residualized_and_rank_matched_erasures(self):
        import csv
        import json
        import subprocess
        import tempfile

        from eegdistill.checkpoints import save_run
        from eegdistill.data import load_ch_names, make_synthetic
        from eegdistill.models import EEGImageModel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg, feat = make_synthetic(root, n_obj_train=4, n_obj_test=6,
                                       feat_dim=8, n_t=4, n_subjects=1)
            checkpoint = root / "checkpoint"
            model = EEGImageModel(range(len(load_ch_names(eeg))), 4, 8, "linear", 5)
            save_run(model, checkpoint, {"time_window": [0, 4], "projector": "linear",
                                          "feature_dim": 5})
            annotations = root / "annotations.csv"
            with annotations.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["split", "index", "image_id", "edge_density", "category"])
                for split, size in [("train", 40), ("test", 6)]:
                    for i in range(size):
                        writer.writerow([split, i, f"{split}/{i}", (i * 7) % 5, "ABC"[i % 3]])
            proc = subprocess.run([
                sys.executable, str(ROOT / "mi_erasure.py"), "--eeg-dir", eeg,
                "--feat-dir", feat, "--annotations", str(annotations),
                "--run", "1", str(checkpoint), "--descriptor", "edge_density",
                "--category", "category", "--residualize", "edge_density",
                "--rank-matched-category", "--controls", "2",
            ], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["derived"]["edge_density_resid_category"]["from"],
                             "edge_density")
            self.assertEqual(report["derived"]["category_rank1"]["kind"], "rank_matched")
            edits = report["subjects"]["1"]["views"]["8"]["interventions"]
            self.assertEqual(edits["category_rank1"]["rank"], 1)
            self.assertEqual(edits["category"]["rank"], 2)
            self.assertIn("edge_density_resid_category", report["summary"]["8"]["effects"])


if __name__ == "__main__":
    unittest.main()
