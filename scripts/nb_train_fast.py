"""Run NeuroBridge's train.py with its EEG smoothing vectorized.

NeuroBridge's RandomSmooth loops over channels and time points in Python, which takes
~12 min per 63-channel subject with --frozen_eeg_prior and far longer for multi-subject
teachers. This replaces RandomSmooth.__call__ with an equivalent numpy version (same kernel,
same per-channel probability, edge-truncated moving average) and then runs train.py
unchanged. The random draws differ from the original, so results match it statistically,
not bit for bit. Usage (from inside the NeuroBridge checkout):
    python /path/to/nb_train_fast.py <train.py args>
"""
import os
import runpy
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from module import eeg_augmentation as aug  # noqa: E402


def _smooth(self, eeg_data):
    ch, t = eeg_data.shape
    h = self.kernel_size // 2
    lo = np.clip(np.arange(t) - h, 0, t)
    hi = np.clip(np.arange(t) + h + 1, 0, t)
    cs = np.concatenate([np.zeros((ch, 1)), np.cumsum(eeg_data, axis=1, dtype=np.float64)], axis=1)
    sm = ((cs[:, hi] - cs[:, lo]) / (hi - lo)).astype(eeg_data.dtype)
    pick = np.random.rand(ch) < self.smooth_prob
    out = np.copy(eeg_data)
    out[pick] = sm[pick]
    return out


aug.RandomSmooth.__call__ = _smooth
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
