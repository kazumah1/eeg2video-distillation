"""Save and load models in NeuroBridge's checkpoint format.

A run directory holds `train_config.json` and `checkpoint_last.pth`
(NeuroBridge's train.py writes exactly these when run with --save_weights).
Our own students are saved the same way, so any student can later serve as
a teacher (e.g. the born-again control).
"""
import json
import os

import torch

from .models import EEGImageModel


def save_run(model, run_dir, config):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    torch.save({
        "model_state_dict": model.encoder.state_dict(),
        "eeg_projector_state_dict": model.eeg_projector.state_dict(),
        "img_projector_state_dict": model.img_projector.state_dict(),
    }, os.path.join(run_dir, "checkpoint_last.pth"))


def load_run(run_dir, ch_names, n_samples, feat_dim, device="cpu", checkpoint="checkpoint_last.pth"):
    """Rebuild an EEGImageModel from a NeuroBridge or eegdistill run directory."""
    with open(os.path.join(run_dir, "train_config.json")) as f:
        cfg = json.load(f)
    enc = cfg.get("eeg_encoder_type", "EEGProject")
    if enc != "EEGProject":
        raise NotImplementedError(f"teacher encoder '{enc}' not supported yet (EEGProject only)")
    chans = cfg.get("channel_names") or cfg.get("selected_channels") or []
    if not chans:
        chans = list(ch_names)
    idx = [ch_names.index(c) for c in chans]
    tw = cfg.get("time_window", [0, n_samples])
    if tw[1] - tw[0] != n_samples:
        raise ValueError(f"teacher time window {tw} does not match data ({n_samples} samples)")
    model = EEGImageModel(idx, n_samples, feat_dim,
                          projector=cfg.get("projector", "direct"),
                          proj_dim=cfg.get("feature_dim", 512))
    ckpt = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(ckpt["model_state_dict"])
    model.eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    model.img_projector.load_state_dict(ckpt["img_projector_state_dict"])
    return model.to(device).eval(), cfg, chans
