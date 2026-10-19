"""One channel-agnostic student for every montage, distilled from the full-cap teacher.

The student always takes all 63 input slots. Each training trial gets a random channel
mask: missing channels are zeroed and a learned mask embedding (added to the first hidden
layer) tells the network which channels are present. The teacher sees the full cap. At
test time the same weights are evaluated on each montage, and one CSV row is written per
montage, tagged "agnostic/<montage>-<condition>[-init]|seed<n>", so `sweep.py summary`
compares it with the dedicated per-montage students.

Montages seen in training: all, nested32, nested16, crown8, occipital8, plus random
subsets of 4-63 channels. Held out (never a named training montage): posterior17,
crown_po2, crown_ant6.

    python train_agnostic.py --teacher runs/nb_s08_all --condition response --seed 0
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from eegdistill import losses  # noqa: E402
from eegdistill.checkpoints import load_run  # noqa: E402
from eegdistill.data import ThingsEEG, random_smooth  # noqa: E402
from eegdistill.evaluate import evaluate  # noqa: E402
from eegdistill.models import EEGImageModel  # noqa: E402
from eegdistill.montages import resolve  # noqa: E402

TRAIN_MONTAGES = ["all", "nested32", "nested16", "crown8", "occipital8"]
EVAL_MONTAGES = TRAIN_MONTAGES + ["posterior17", "crown_po2", "crown_ant6"]
PRESETS = {"none": dict(w_emb=0.0, w_kl=0.0), "response": dict(w_emb=1.0, w_kl=1.0)}


class AgnosticModel(nn.Module):
    """EEGImageModel over all channels + a mask embedding on the first hidden layer."""

    def __init__(self, n_ch, n_samples, feat_dim, projector, proj_dim):
        super().__init__()
        self.base = EEGImageModel(list(range(n_ch)), n_samples, feat_dim, projector, proj_dim)
        self.mask_embed = nn.Linear(n_ch, feat_dim, bias=False)
        nn.init.zeros_(self.mask_embed.weight)  # full mask + teacher init == the teacher
        self.encode_img = self.base.encode_img
        self.img_projector = self.base.img_projector

    def encode_eeg(self, x, mask, return_hidden=False):
        enc = self.base.encoder
        xm = (x * mask[:, :, None]).reshape(x.shape[0], enc.input_dim)
        h = enc.model[0](xm) + self.mask_embed(mask) - self.mask_embed(torch.ones_like(mask))
        out = enc.model[2](enc.model[1](h))
        z = self.base.eeg_projector(out)
        return (z, h) if return_hidden else z


class FixedMask(nn.Module):
    """View of an AgnosticModel with one montage, so evaluate() can use it unchanged."""

    def __init__(self, model, mask):
        super().__init__()
        self.model = model
        self.register_buffer("mask", mask)

    def encode_eeg(self, x, return_hidden=False):
        return self.model.encode_eeg(x, self.mask.expand(x.shape[0], -1), return_hidden)

    def encode_img(self, f):
        return self.model.encode_img(f)


def sample_masks(n, montage_masks, n_ch, p_random, gen):
    """Per-trial masks: a named training montage, or a random subset of 4..n_ch channels."""
    masks = torch.zeros(n, n_ch)
    use_random = torch.rand(n, generator=gen) < p_random
    pick = torch.randint(len(montage_masks), (n,), generator=gen)
    for i in range(n):
        if use_random[i]:
            k = int(torch.randint(4, n_ch + 1, (1,), generator=gen))
            masks[i, torch.randperm(n_ch, generator=gen)[:k]] = 1.0
        else:
            masks[i] = montage_masks[pick[i]]
    return masks


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg_dir", default="data/things_eeg/preprocessed_eeg")
    p.add_argument("--feat_dir", default="data/things_eeg/image_feature/RN50")
    p.add_argument("--aug_dir", default="data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic")
    p.add_argument("--subject", type=int, default=8)
    p.add_argument("--teacher", required=True)
    p.add_argument("--condition", default="response", choices=sorted(PRESETS))
    p.add_argument("--init_from_teacher", type=int, default=0, help="1 = start from the teacher's weights (same shape)")
    p.add_argument("--p_random", type=float, default=0.5, help="share of trials with a random channel subset")
    p.add_argument("--kd_tau", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eeg_smooth", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--results_csv", default="results/results.csv")
    p.add_argument("--tag", default="", help="ignored except as a prefix check; rows are tagged per montage")
    p.add_argument("--runs_dir", default="runs")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--device", default=dev)
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    dev = torch.device(a.device)
    w = PRESETS[a.condition]
    t0 = time.time()

    aug_dir = a.aug_dir if a.aug_dir and os.path.isdir(a.aug_dir) else None
    train = ThingsEEG(a.eeg_dir, a.feat_dir, [a.subject], "train", average=True, aug_dir=aug_dir)
    test = ThingsEEG(a.eeg_dir, a.feat_dir, [a.subject], "test", average=True, aug_dir=aug_dir)
    if a.eeg_smooth:
        train.eeg = random_smooth(train.eeg, seed=a.seed)
    ch = train.ch_names
    teacher, tcfg, t_chans = load_run(a.teacher, ch, train.n_samples, train.feat_dim, device=dev)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    def mask_of(m):
        _, idx = resolve(m, ch)
        v = torch.zeros(len(ch))
        v[idx] = 1.0
        return v

    train_masks = [mask_of(m) for m in TRAIN_MONTAGES]
    projector, proj_dim = tcfg.get("projector", "linear"), tcfg.get("feature_dim", 512)
    model = AgnosticModel(len(ch), train.n_samples, train.feat_dim, projector, proj_dim)
    if a.init_from_teacher:
        if len(t_chans) != len(ch):
            raise SystemExit("--init_from_teacher needs a full-cap teacher")
        model.base.encoder.load_state_dict(teacher.encoder.state_dict())
        model.base.eeg_projector.load_state_dict(teacher.eeg_projector.state_dict())
    model.base.img_projector.load_state_dict(teacher.img_projector.state_dict())
    for p_ in model.base.img_projector.parameters():
        p_.requires_grad_(False)
    model = model.to(dev)
    task = losses.ContrastiveLoss(init_temperature=tcfg.get("init_temperature", 0.07),
                                  softplus=tcfg.get("softplus", True)).to(dev)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                            lr=a.lr, weight_decay=a.weight_decay)

    n = len(train)
    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, nb = {"task": 0.0, "emb": 0.0, "kl": 0.0}, 0
        for i in range(0, n - a.batch_size + 1, a.batch_size):
            idx = perm[i:i + a.batch_size]
            x = train.eeg[idx].to(dev)
            f_img = train.train_image_features(idx, generator=gen).to(dev)
            mask = sample_masks(len(idx), train_masks, len(ch), a.p_random, gen).to(dev)
            z_s = model.encode_eeg(x, mask)
            loss_task = task(z_s, model.encode_img(f_img))
            loss = loss_task
            tot["task"] += loss_task.item()
            if w["w_emb"] or w["w_kl"]:
                with torch.no_grad():
                    z_t = teacher.encode_eeg(x)
                    z_img_t = teacher.encode_img(f_img)
                l_e = losses.response_embedding_loss(z_s, z_t)
                l_k = losses.response_kl_loss(z_s, z_t, z_img_t, a.kd_tau)
                loss = loss + w["w_emb"] * l_e + w["w_kl"] * l_k
                tot["emb"] += l_e.item()
                tot["kl"] += l_k.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            nb += 1
        msg = " ".join(f"{k}={v / nb:.4f}" for k, v in tot.items() if v)
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            r = evaluate(FixedMask(model, mask_of("crown8").to(dev)), test, dev)
            msg += f" | crown8 top1 {r['top1']:.2f}"
        print(f"epoch {epoch:3d} {msg}", flush=True)

    minutes = round((time.time() - t0) / 60, 1)
    suffix = f"{a.condition}{'-init' if a.init_from_teacher else ''}"
    rows = []
    for m in EVAL_MONTAGES:
        res = evaluate(FixedMask(model, mask_of(m).to(dev)), test, dev, teacher=teacher)
        rows.append({
            "time": time.strftime("%Y-%m-%d %H:%M"), "run": f"agnostic_{m}_{suffix}_seed{a.seed}",
            "tag": f"agnostic/{m}-{suffix}|seed{a.seed}", "subject": a.subject, "montage": m,
            "n_channels": int(mask_of(m).sum()), "teacher": os.path.basename(os.path.normpath(a.teacher)),
            "teacher_channels": len(t_chans), "condition": f"agnostic-{a.condition}", **w,
            "w_feat": 0.0, "w_rel": 0.0, "kd_tau": a.kd_tau, "img_proj": "teacher",
            "train_average": 1, "test_average": 1, "eeg_smooth": a.eeg_smooth,
            "init_from_teacher": a.init_from_teacher, "img_feats": "augmean" if aug_dir else "clean",
            "epochs": a.epochs, "seed": a.seed,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, "minutes": minutes,
        })
        print(f"{m:12s} top1 {res['top1']:.1f} top5 {res['top5']:.1f}")
    os.makedirs(os.path.dirname(a.results_csv) or ".", exist_ok=True)
    exists = os.path.exists(a.results_csv)
    if exists:  # keep the existing header so rows line up
        with open(a.results_csv) as f:
            fields = next(csv.reader(f))
    else:
        fields = list(rows[0])
    with open(a.results_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(json.dumps({r["montage"]: r["top1"] for r in rows}))


if __name__ == "__main__":
    main()
