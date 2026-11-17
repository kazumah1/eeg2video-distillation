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


class ShuffledControlTest(unittest.TestCase):
    def test_shuffles_move_whole_train_concepts_and_keep_test_labels_seen(self):
        from eegdistill.derived import add_derived_labels

        groups = np.repeat([0, 1, 2, 3], 2)
        train = {"edges": np.arange(8.0), "cat": np.repeat(list("AABB"), 2)}
        test = {"edges": np.array([10.0, 20.0, 30.0]), "cat": np.array(["A", "B", "B"])}
        new_train, new_test, derived = add_derived_labels(
            np.zeros((8, 3)), train, test, category="cat", rank_matched=False,
            shuffles=4, train_groups=groups, seed=0)
        blocks = {(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)}
        for k in range(4):
            edges = new_train[f"edges_shuf{k}"]
            self.assertEqual({tuple(edges[groups == g]) for g in range(4)}, blocks)
            cats = new_train[f"cat_shuf{k}"]
            self.assertTrue(all(len(set(cats[groups == g])) == 1 for g in range(4)))
            self.assertEqual(sorted(cats), sorted(train["cat"]))
            self.assertEqual(sorted(new_test[f"edges_shuf{k}"]), [10.0, 20.0, 30.0])
            self.assertEqual(sorted(new_test[f"cat_shuf{k}"]), ["A", "B", "B"])
            self.assertEqual(derived[f"edges_shuf{k}"], {"kind": "shuffled", "from": "edges"})
        # Not every shuffle is the identity.
        self.assertFalse(all(np.array_equal(new_train[f"edges_shuf{k}"], train["edges"])
                             for k in range(4)))

    def test_unequal_concept_sizes_are_rejected(self):
        from eegdistill.derived import add_derived_labels

        with self.assertRaises(ValueError):
            add_derived_labels(np.zeros((3, 2)), {"e": np.arange(3.0), "c": np.array(list("AAB"))},
                               {"e": np.arange(2.0), "c": np.array(list("AB"))}, category="c",
                               rank_matched=False, shuffles=1, train_groups=np.array([0, 0, 1]))


class JointErasureTest(unittest.TestCase):
    def test_joint_label_stacks_descriptors_and_erases_their_shared_span(self):
        from eegdistill.derived import add_derived_labels
        from eegdistill.erasure import evaluate_erasure

        rng = np.random.default_rng(0)
        z = rng.normal(size=(60, 6))
        test_z = rng.normal(size=(12, 6))
        # Two descriptors read from dims 0 and 1; a third, unrelated, from dim 4.
        train = {"a": z[:, 0], "b": z[:, 1], "c": z[:, 4], "cat": np.repeat(list("XY"), 30)}
        test = {"a": test_z[:, 0], "b": test_z[:, 1], "c": test_z[:, 4],
                "cat": np.repeat(list("XY"), 6)}
        train, test, derived = add_derived_labels(
            z, train, test, category="cat", rank_matched=False, joint={"ab": ["a", "b"]})
        self.assertEqual(derived["ab"], {"kind": "joint", "of": ["a", "b"]})
        np.testing.assert_array_equal(train["ab"], np.stack([z[:, 0], z[:, 1]], axis=1))
        report = evaluate_erasure(z, test_z, test_z, train, test,
                                  categorical=("cat",), controls=2)
        joint = report["interventions"]["ab"]
        self.assertEqual(joint["rank"], 2)
        base = report["baseline"]["probes"]
        # Joint score is the mean R^2 of its columns; probes see unit-normalized rows.
        self.assertAlmostEqual(base["ab"]["r2"], (base["a"]["r2"] + base["b"]["r2"]) / 2)
        self.assertGreater(base["ab"]["r2"], 0.85)
        probes = joint["target"]["probes"]
        self.assertLess(probes["ab"]["r2"], 0.05)  # both columns removed
        self.assertLess(probes["a"]["r2"], 0.05)
        self.assertLess(probes["b"]["r2"], 0.05)
        self.assertGreater(probes["c"]["r2"], 0.85)  # unrelated descriptor kept

    def test_joint_label_gets_concept_shuffled_copies_that_keep_rows_together(self):
        from eegdistill.derived import add_derived_labels

        groups = np.repeat([0, 1, 2, 3], 2)
        train = {"a": np.arange(8.0), "b": np.arange(8.0) * 10, "cat": np.repeat(list("AABB"), 2)}
        test = {"a": np.array([1.0, 2.0]), "b": np.array([10.0, 20.0]), "cat": np.array(list("AB"))}
        new_train, new_test, derived = add_derived_labels(
            np.zeros((8, 3)), train, test, category="cat", rank_matched=False,
            joint={"ab": ["a", "b"]}, shuffles=2, train_groups=groups)
        shuffled = new_train["ab_shuf0"]
        self.assertEqual(shuffled.shape, (8, 2))
        np.testing.assert_array_equal(shuffled[:, 1], shuffled[:, 0] * 10)  # rows intact
        np.testing.assert_array_equal(new_test["ab_shuf0"][:, 1], new_test["ab_shuf0"][:, 0] * 10)
        self.assertEqual(derived["ab_shuf1"], {"kind": "shuffled", "from": "ab"})


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
                writer.writerow(["split", "index", "image_id", "edge_density", "contrast", "category"])
                for split, size in [("train", 40), ("test", 6)]:
                    for i in range(size):
                        writer.writerow([split, i, f"{split}/{i}", (i * 7) % 5, (i * 3) % 4, "ABC"[i % 3]])
            proc = subprocess.run([
                sys.executable, str(ROOT / "mi_erasure.py"), "--eeg-dir", eeg,
                "--feat-dir", feat, "--annotations", str(annotations),
                "--run", "1", str(checkpoint), "--descriptor", "edge_density",
                "--category", "category", "--residualize", "edge_density",
                "--rank-matched-category", "--shuffled-controls", "2", "--controls", "2",
                "--descriptor", "contrast", "--joint", "lowlevel=edge_density,contrast",
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
            self.assertEqual(report["derived"]["category_shuf1"], {"kind": "shuffled",
                                                                 "from": "category"})
            self.assertEqual(report["derived"]["lowlevel"],
                             {"kind": "joint", "of": ["edge_density", "contrast"]})
            self.assertEqual(edits["lowlevel"]["rank"], 2)
            self.assertEqual(report["derived"]["lowlevel_shuf0"]["from"], "lowlevel")
            # Shuffled category stays categorical: probed by balanced accuracy.
            self.assertIn("balanced_accuracy",
                          edits["category_shuf0"]["target"]["probes"]["category_shuf0"])


if __name__ == "__main__":
    unittest.main()
