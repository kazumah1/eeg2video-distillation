"""Behavior tests for the frozen-embedding analysis and checkpoint runner."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ErasureTest(unittest.TestCase):
    def test_target_removal_preserves_independent_property_and_matches_control(self):
        from eegdistill.erasure import evaluate_erasure

        # Factorial embeddings separate two independent annotations by design.
        x = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        train = np.tile(x, (20, 1))
        report = evaluate_erasure(
            train, x, x, {"edges": train[:, 0], "contrast": train[:, 1]},
            {"edges": x[:, 0], "contrast": x[:, 1]}, controls=4,
        )
        edge = report["interventions"]["edges"]
        self.assertEqual(report["baseline"]["top1"], 1.0)
        self.assertGreater(report["baseline"]["probes"]["edges"]["r2"], 0.99)
        self.assertAlmostEqual(edge["target"]["probes"]["edges"]["r2"], 0.0, places=8)
        self.assertGreater(edge["target"]["probes"]["contrast"]["r2"], 0.99)
        self.assertEqual(edge["target"]["top1"], 0.0)  # Tied best scores count as misses.
        self.assertEqual(edge["rank"], 1)
        self.assertEqual(len(edge["controls"]), 4)
        for control in edge["controls"]:
            self.assertLess(control["max_l2_matching_error"], 1e-12)

    def test_checkpoint_runner_keeps_subjects_paired_and_reports_repeat_views(self):
        from eegdistill.checkpoints import save_run
        from eegdistill.data import load_ch_names, make_synthetic
        from eegdistill.models import EEGImageModel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg, feat = make_synthetic(root, n_obj_train=4, n_obj_test=6,
                                       feat_dim=8, n_t=4, n_subjects=2)
            checkpoint = root / "checkpoint"
            names = load_ch_names(eeg)
            model = EEGImageModel(range(len(names)), 4, 8, "linear", 5)
            save_run(model, checkpoint, {"time_window": [0, 4], "projector": "linear",
                                          "feature_dim": 5})
            annotations = root / "annotations.csv"
            with annotations.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["split", "index", "image_id", "edge_density", "category"])
                for split, size in [("train", 40), ("test", 6)]:
                    for i in range(size):
                        writer.writerow([split, i, f"{split}/{i}", i % 3, "AB"[i % 2]])
            command = [
                sys.executable, str(ROOT / "mi_erasure.py"), "--eeg-dir", eeg,
                "--feat-dir", feat, "--annotations", str(annotations),
                "--run", "1", str(checkpoint), "--run", "2", str(checkpoint),
                "--descriptor", "edge_density", "--category", "category",
                "--repeats", "1", "4", "all", "--controls", "2",
            ]
            proc = subprocess.run(command, capture_output=True, text=True,
                                  env={**os.environ, "OMP_NUM_THREADS": "1"})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(set(report["subjects"]), {"1", "2"})
            self.assertEqual(set(report["subjects"]["1"]["views"]), {"1", "4", "8"})
            self.assertEqual(report["summary"]["4"]["effects"]["edge_density"]["top1"]["n_subjects"], 2)
            self.assertIn("edge_density_minus_category", report["summary"]["4"][
                "paired_contrasts"])
            # Reject a reused test image even with a valid array index.
            annotations.write_text(annotations.read_text().replace("test/0", "train/0"))
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("duplicate image_id", proc.stderr)


class SizeMatchedShuffleTest(unittest.TestCase):
    def test_size_matched_copy_moves_each_image_as_far_as_the_real_erasure(self):
        from eegdistill.erasure import evaluate_erasure

        rng = np.random.default_rng(1)
        z = rng.normal(size=(80, 6))
        test_z = rng.normal(size=(20, 6))
        z[:, 0] *= 5  # the real label reads a high-variance axis: a big edit
        test_z[:, 0] *= 5
        train = {"real": z[:, 0], "noise": rng.normal(size=80)}
        test = {"real": test_z[:, 0], "noise": rng.normal(size=20)}
        report = evaluate_erasure(z, test_z, test_z, train, test, controls=2,
                                  size_matched={"noise": "real"})
        real = report["interventions"]["real"]
        shuffled = report["interventions"]["noise"]
        matched = shuffled["size_matched"]
        self.assertLess(shuffled["target"]["mean_l2"], 0.5 * real["target"]["mean_l2"])
        self.assertAlmostEqual(matched["mean_l2"], real["target"]["mean_l2"])
        self.assertLess(matched["max_l2_matching_error"], 1e-9)  # per image, not on average
        self.assertNotIn("size_matched", real)


if __name__ == "__main__":
    unittest.main()
