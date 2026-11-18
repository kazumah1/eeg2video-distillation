"""Behavior tests for scripts/erasure_all_subjects.sh (dry-run command plan)."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dry_run(out, *subjects, **env):
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/erasure_all_subjects.sh"), *map(str, subjects)],
        env={**os.environ, "ANNOT": "/x/annotations.csv", "OUT": str(out), "DRY_RUN": "1",
             "DATA": str(Path(out) / "data"), "RUNS": str(Path(out) / "runs"), **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class MontageDriverTest(unittest.TestCase):
    def test_default_runs_the_63_channel_teacher_only(self):
        with tempfile.TemporaryDirectory() as out:
            plan = dry_run(out, 3)
        self.assertIn(f"{out}/runs/nb_s03_all", plan)
        self.assertIn(f"{out}/sub-03.json", plan)
        self.assertNotIn("train_student.py", plan)

    def test_montages_train_no_kd_students_and_write_one_report_per_montage(self):
        with tempfile.TemporaryDirectory() as out:
            plan = dry_run(out, 3, MONTAGES="crown8 occipital8")
        for m in ("crown8", "occipital8"):
            self.assertIn(f"--montage {m} --condition none --subject 3 --save_as nb_s03_{m}", plan)
            self.assertIn(f"{out}/runs/nb_s03_{m} ", plan)
            self.assertIn(f"{out}/{m}/sub-03.json", plan)
        # Students borrow the 63-channel teacher's frozen image projector.
        self.assertIn(f"--teacher {out}/runs/nb_s03_all", plan)
        self.assertNotIn(f"{out}/sub-03.json", plan)
        # EEG is downloaded once and removed once, after every montage.
        self.assertEqual(plan.count("download_things_eeg.py"), 1)
        self.assertEqual(plan.count("rm -rf"), 1)

    def test_finished_montages_are_skipped(self):
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(f"{out}/crown8")
            Path(f"{out}/crown8/sub-03.json").write_text("{}")
            plan = dry_run(out, 3, MONTAGES="crown8 occipital8")
        self.assertNotIn("--montage crown8", plan)
        self.assertIn("--montage occipital8", plan)


if __name__ == "__main__":
    unittest.main()
