"""THINGS-EEG loading in the NeuroBridge preprocessed format.

Expected layout (what NeuroBridge's preprocess_eeg.py / its Hugging Face copy produce):

    <eeg_dir>/info.json                 {"ch_names": [...63 names...], ...}
    <eeg_dir>/sub-08/train.npy          (1654 objects, 10 images, 4 reps, 63 ch, 250 samples)
    <eeg_dir>/sub-08/test.npy           (200 objects, 1 image, 80 reps, 63 ch, 250 samples)
    <feat_dir>/image_train.npy          (1654, 10, D)
    <feat_dir>/image_test.npy           (200, 1, D)

The dataset always returns ALL channels. Teacher and student pick their own
channel subsets inside the training loop, so both see exactly the same trial.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def load_ch_names(eeg_dir):
    with open(os.path.join(eeg_dir, "info.json")) as f:
        return json.load(f)["ch_names"]


class ThingsEEG(Dataset):
    """Flat (trial, image-feature) pairs for one or more subjects.

    split:   'train' or 'test'
    average: True  -> average the repetitions (4 train / 80 test) into one trial
             False -> every repetition is its own sample (single-trial)
    """

    def __init__(self, eeg_dir, feat_dir, subjects, split="train", average=True,
                 time_window=(0, 250), dtype=np.float32):
        assert split in ("train", "test")
        self.ch_names = load_ch_names(eeg_dir)
        feats = np.load(os.path.join(feat_dir, f"image_{split}.npy")).astype(np.float32)
        n_obj, n_img, feat_dim = feats.shape
        self.feat_dim = feat_dim
        self.image_features = torch.from_numpy(feats.reshape(n_obj * n_img, feat_dim))

        eeg_parts, img_idx_parts, subj_parts = [], [], []
        for s in subjects:
            arr = np.load(os.path.join(eeg_dir, f"sub-{s:02d}", f"{split}.npy"), mmap_mode="r")
            assert arr.shape[:2] == (n_obj, n_img), f"EEG {arr.shape} vs features {feats.shape}"
            t0, t1 = time_window
            arr = arr[..., t0:t1]
            if average:
                x = np.asarray(arr.mean(axis=2), dtype=dtype)           # (obj, img, C, T)
                x = x.reshape(n_obj * n_img, *x.shape[-2:])
                img_idx = np.arange(n_obj * n_img)
            else:
                n_rep = arr.shape[2]
                x = np.asarray(arr, dtype=dtype).reshape(n_obj * n_img * n_rep, *arr.shape[-2:])
                img_idx = np.repeat(np.arange(n_obj * n_img), n_rep)
            eeg_parts.append(x)
            img_idx_parts.append(img_idx)
            subj_parts.append(np.full(len(img_idx), s))

        self.eeg = torch.from_numpy(np.concatenate(eeg_parts))
        self.img_idx = torch.from_numpy(np.concatenate(img_idx_parts)).long()
        self.subject = torch.from_numpy(np.concatenate(subj_parts)).long()
        self.n_channels, self.n_samples = self.eeg.shape[1:]

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, i):
        j = self.img_idx[i]
        return self.eeg[i], self.image_features[j], j


def make_synthetic(root, n_obj_train=40, n_obj_test=10, feat_dim=64, n_ch=63, n_t=250, seed=0):
    """Tiny fake dataset in the same layout, for smoke tests without downloads."""
    from .montages import CROWN_8, POSTERIOR_17, NESTED_32
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(NESTED_32 + POSTERIOR_17 + CROWN_8))
    names += [f"X{i}" for i in range(n_ch - len(names))]
    eeg_dir, feat_dir = os.path.join(root, "eeg"), os.path.join(root, "feat")
    os.makedirs(os.path.join(eeg_dir, "sub-01"), exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)
    with open(os.path.join(eeg_dir, "info.json"), "w") as f:
        json.dump({"ch_names": names}, f)
    mix = rng.normal(size=(feat_dim, n_ch * n_t)).astype(np.float32) / np.sqrt(feat_dim)
    for split, n_obj, n_img, n_rep in [("train", n_obj_train, 10, 4), ("test", n_obj_test, 1, 8)]:
        f_ = rng.normal(size=(n_obj, n_img, feat_dim)).astype(np.float32)
        sig = (f_.reshape(-1, feat_dim) @ mix).reshape(n_obj, n_img, 1, n_ch, n_t)
        x = sig + rng.normal(scale=2.0, size=(n_obj, n_img, n_rep, n_ch, n_t)).astype(np.float32)
        np.save(os.path.join(eeg_dir, "sub-01", f"{split}.npy"), x.astype(np.float32))
        np.save(os.path.join(feat_dir, f"image_{split}.npy"), f_)
    return eeg_dir, feat_dir
