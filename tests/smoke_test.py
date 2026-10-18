"""End-to-end smoke test on a tiny synthetic dataset (CPU, ~1 minute).

    python tests/smoke_test.py

Builds fake data in the real layout, saves a fake 63-channel "teacher" in
NeuroBridge checkpoint format, then runs every student condition plus the
born-again path, and checks the results CSV.
"""
import csv
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from eegdistill.checkpoints import save_run  # noqa: E402
from eegdistill.data import ThingsEEG, load_ch_names, make_synthetic  # noqa: E402
from eegdistill.models import EEGImageModel  # noqa: E402


def run(args):
    cmd = [sys.executable, os.path.join(ROOT, "train_student.py")] + args
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:], out.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    return out.stdout


def main():
    tmp = tempfile.mkdtemp()
    eeg_dir, feat_dir = make_synthetic(tmp)
    ch = load_ch_names(eeg_dir)
    ds = ThingsEEG(eeg_dir, feat_dir, [1], "train", average=False)
    assert ds.eeg.shape == (40 * 10 * 4, 63, 250), ds.eeg.shape
    ds_avg = ThingsEEG(eeg_dir, feat_dir, [1], "test", average=True)
    assert ds_avg.eeg.shape == (10, 63, 250), ds_avg.eeg.shape

    teacher = EEGImageModel(list(range(63)), 250, ds.feat_dim, "linear", 32)
    tdir = os.path.join(tmp, "runs", "fake_teacher")
    save_run(teacher, tdir, {"eeg_encoder_type": "EEGProject", "selected_channels": [],
                             "projector": "linear", "feature_dim": 32, "time_window": [0, 250],
                             "init_temperature": 0.07, "softplus": True})
    common = ["--eeg_dir", eeg_dir, "--feat_dir", feat_dir, "--subject", "1", "--teacher", tdir,
              "--epochs", "2", "--batch_size", "64", "--eval_every", "1", "--device", "cpu",
              "--runs_dir", os.path.join(tmp, "runs"), "--results_csv", os.path.join(tmp, "res.csv")]
    for cond in ["none", "response", "feature", "relational", "all"]:
        run(common + ["--montage", "crown8", "--condition", cond])
        print("ok", cond)
    run(common + ["--montage", "crown8", "--condition", "none", "--save_as", "ba"])
    run(common[:common.index("--teacher")] + ["--teacher", os.path.join(tmp, "runs", "ba")] +
        common[common.index("--teacher") + 2:] + ["--montage", "crown8", "--condition", "response", "--tag", "born_again"])
    print("ok born-again")
    run(common + ["--montage", "nested16", "--condition", "response", "--train_average", "0",
                  "--test_average", "0", "--img_proj", "own"])
    print("ok single-trial + own projector")
    rows = list(csv.DictReader(open(os.path.join(tmp, "res.csv"))))
    assert len(rows) == 8, len(rows)
    assert rows[-2]["teacher"] == "ba" and rows[-2]["teacher_channels"] == "8"
    print(f"PASS: {len(rows)} runs logged; example row: top1={rows[0]['top1']} "
          f"teacher_top1={rows[0]['teacher_top1']} fid_cos={rows[0]['fid_cos']}")


if __name__ == "__main__":
    main()
