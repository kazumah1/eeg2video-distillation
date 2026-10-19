"""Train an EEG-to-image student, with or without distillation from a teacher.

Examples
--------
No-distillation baseline (lambda = 0), Crown montage, using the teacher only
for its frozen image projector and for evaluation:
    python train_student.py --teacher runs/nb_sub08_all --montage crown8 --condition none

Response distillation:
    python train_student.py --teacher runs/nb_sub08_all --montage crown8 --condition response

Born-again control: first save an 8-channel student, then use it as the teacher:
    python train_student.py --teacher runs/nb_sub08_all --montage crown8 --condition none --save_as ba_teacher
    python train_student.py --teacher runs/ba_teacher --montage crown8 --condition response --tag born_again

Every run appends one row to --results_csv.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from eegdistill import losses  # noqa: E402
from eegdistill.checkpoints import load_run, save_run  # noqa: E402
from eegdistill.data import ThingsEEG, random_smooth  # noqa: E402
from eegdistill.evaluate import evaluate  # noqa: E402
from eegdistill.models import EEGImageModel, init_from_teacher  # noqa: E402
from eegdistill.montages import resolve  # noqa: E402

PRESETS = {  # weights on each distillation term
    "none":       dict(w_emb=0.0, w_kl=0.0, w_feat=0.0, w_rel=0.0),
    "response":   dict(w_emb=1.0, w_kl=1.0, w_feat=0.0, w_rel=0.0),
    "emb":        dict(w_emb=1.0, w_kl=0.0, w_feat=0.0, w_rel=0.0),
    "kl":         dict(w_emb=0.0, w_kl=1.0, w_feat=0.0, w_rel=0.0),
    "feature":    dict(w_emb=0.0, w_kl=0.0, w_feat=1.0, w_rel=0.0),
    "relational": dict(w_emb=0.0, w_kl=0.0, w_feat=0.0, w_rel=1.0),
    "all":        dict(w_emb=1.0, w_kl=1.0, w_feat=1.0, w_rel=1.0),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg_dir", default="data/things_eeg/preprocessed_eeg")
    p.add_argument("--feat_dir", default="data/things_eeg/image_feature/RN50")
    p.add_argument("--aug_dir", default="data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic",
                   help="augmented image features (what NeuroBridge teachers train on); '' = clean only")
    p.add_argument("--eval_img", default="augmean", choices=["augmean", "augrand", "clean"],
                   help="test image features: mean over augmented copies (default), one fixed random copy, or clean")
    p.add_argument("--subject", type=int, default=8)
    p.add_argument("--teacher", required=True, help="run dir with train_config.json + checkpoint")
    p.add_argument("--teacher_ckpt", default="checkpoint_last.pth")
    p.add_argument("--montage", default="crown8", help="montage name in montages.py")
    p.add_argument("--condition", default="none", choices=sorted(PRESETS))
    for w in ["w_emb", "w_kl", "w_feat", "w_rel"]:
        p.add_argument(f"--{w}", type=float, default=None, help="override the preset weight")
    p.add_argument("--kd_tau", type=float, default=4.0)
    p.add_argument("--init_from_teacher", type=int, default=0,
                   help="1 = start the student from the teacher's weights (shared channels' input "
                        "columns, all later layers, both projectors)")
    p.add_argument("--img_proj", default="teacher", choices=["teacher", "own"],
                   help="teacher: copy and freeze the teacher's image projector (same embedding "
                        "space in every condition); own: train a fresh one")
    p.add_argument("--train_average", type=int, default=1, help="1 = average the 4 train reps")
    p.add_argument("--test_average", type=int, default=1, help="1 = average the 80 test reps")
    p.add_argument("--eeg_smooth", type=int, default=1,
                   help="1 = NeuroBridge's one-time random smoothing of train EEG (matches the teacher's training data)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--save_as", default=None, help="save the student as runs/<name> (reusable as a teacher)")
    p.add_argument("--results_csv", default="results/results.csv")
    p.add_argument("--tag", default="")
    default_dev = ("cuda" if torch.cuda.is_available()
                   else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                   else "cpu")
    p.add_argument("--device", default=default_dev, help="cuda, mps (Apple GPU) or cpu")
    return p.parse_args()


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    a = parse()
    seed_all(a.seed)
    w = dict(PRESETS[a.condition])
    for k in w:
        if getattr(a, k) is not None:
            w[k] = getattr(a, k)
    dev = torch.device(a.device)
    t0 = time.time()

    print(f">>> loading subject {a.subject} (train_average={a.train_average}, test_average={a.test_average})")
    aug_dir = a.aug_dir if a.aug_dir and os.path.isdir(a.aug_dir) else None
    if a.aug_dir and not aug_dir:
        print(f"WARNING: {a.aug_dir} not found; using clean image features (mismatched with an image-aug teacher)")
    train = ThingsEEG(a.eeg_dir, a.feat_dir, [a.subject], "train", average=bool(a.train_average), aug_dir=aug_dir)
    test = ThingsEEG(a.eeg_dir, a.feat_dir, [a.subject], "test", average=bool(a.test_average),
                     aug_dir=aug_dir, eval_img=a.eval_img)
    if aug_dir:
        print(f"    image features: {train.aug_features.shape[0]} augmented copies per train image; eval={a.eval_img}")
    if a.eeg_smooth:
        train.eeg = random_smooth(train.eeg, seed=a.seed)
    print(f"    train {tuple(train.eeg.shape)}  test {tuple(test.eeg.shape)}  feat_dim {train.feat_dim}")

    teacher, tcfg, t_chans = load_run(a.teacher, train.ch_names, train.n_samples, train.feat_dim,
                                      device=dev, checkpoint=a.teacher_ckpt)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    s_chans, s_idx = resolve(a.montage, train.ch_names)
    print(f"    teacher channels {len(t_chans)}  student channels {len(s_chans)}: {s_chans if len(s_chans) <= 17 else '...'}")

    projector = tcfg.get("projector", "direct")
    proj_dim = tcfg.get("feature_dim", 512)
    student = EEGImageModel(s_idx, train.n_samples, train.feat_dim, projector, proj_dim).to(dev)
    if a.init_from_teacher:
        shared = init_from_teacher(student, teacher, s_chans, t_chans)
        print(f"    init from teacher: {shared}/{len(s_chans)} student channels inherited")
    if a.img_proj == "teacher":
        student.img_projector.load_state_dict(teacher.img_projector.state_dict())
        for p_ in student.img_projector.parameters():
            p_.requires_grad_(False)

    task = losses.ContrastiveLoss(init_temperature=tcfg.get("init_temperature", 0.07),
                                  softplus=tcfg.get("softplus", True)).to(dev)
    params = [p_ for p_ in student.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=a.weight_decay)

    needs_teacher = any(v > 0 for v in w.values())
    n = len(train)
    curve = []
    for epoch in range(1, a.epochs + 1):
        student.train()
        perm = torch.randperm(n)
        sums = {"task": 0.0, "emb": 0.0, "kl": 0.0, "feat": 0.0, "rel": 0.0}
        n_batches = 0
        for i in range(0, n - a.batch_size + 1, a.batch_size):  # drop_last
            idx = perm[i:i + a.batch_size]
            x = train.eeg[idx].to(dev, non_blocking=True)
            f_img = train.train_image_features(idx).to(dev, non_blocking=True)
            z_s, h_s = student.encode_eeg(x, return_hidden=True)
            z_img = student.encode_img(f_img)
            loss_task = task(z_s, z_img)
            loss = loss_task
            parts = {"task": loss_task.item()}
            if needs_teacher:
                with torch.no_grad():
                    z_t, h_t = teacher.encode_eeg(x, return_hidden=True)
                    z_img_t = teacher.encode_img(f_img)
                if w["w_emb"]:
                    l_ = losses.response_embedding_loss(z_s, z_t); loss = loss + w["w_emb"] * l_; parts["emb"] = l_.item()
                if w["w_kl"]:
                    l_ = losses.response_kl_loss(z_s, z_t, z_img_t, a.kd_tau); loss = loss + w["w_kl"] * l_; parts["kl"] = l_.item()
                if w["w_feat"]:
                    l_ = losses.feature_loss(h_s, h_t); loss = loss + w["w_feat"] * l_; parts["feat"] = l_.item()
                if w["w_rel"]:
                    l_ = losses.relational_loss(z_s, z_t); loss = loss + w["w_rel"] * l_; parts["rel"] = l_.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in parts.items():
                sums[k] += v
            n_batches += 1
        msg = " ".join(f"{k}={v / n_batches:.4f}" for k, v in sums.items() if v)
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            r = evaluate(student, test, dev)
            curve.append({"epoch": epoch, "top1": r["top1"], "top5": r["top5"]})
            msg += f" | test top1 {r['top1']:.2f} top5 {r['top5']:.2f}"
        print(f"epoch {epoch:3d} {msg}", flush=True)

    # Final-epoch numbers only: picking the best epoch on the test set would leak.
    res = evaluate(student, test, dev, teacher=teacher)
    run_name = a.save_as or f"s{a.subject:02d}_{a.montage}_{a.condition}_seed{a.seed}{'_' + a.tag if a.tag else ''}"
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M"), "run": run_name, "tag": a.tag,
        "subject": a.subject, "montage": a.montage, "n_channels": len(s_chans),
        "teacher": os.path.basename(os.path.normpath(a.teacher)), "teacher_channels": len(t_chans),
        "condition": a.condition, **w, "kd_tau": a.kd_tau, "img_proj": a.img_proj,
        "train_average": a.train_average, "eeg_smooth": a.eeg_smooth,
        "init_from_teacher": a.init_from_teacher, "img_feats": a.eval_img if aug_dir else "clean", "test_average": a.test_average,
        "epochs": a.epochs, "seed": a.seed, **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(json.dumps(row, indent=1))
    os.makedirs(os.path.dirname(a.results_csv) or ".", exist_ok=True)
    new = not os.path.exists(a.results_csv)
    with open(a.results_csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        if new:
            wr.writeheader()
        wr.writerow(row)
    if a.save_as:
        cfg = {"eeg_encoder_type": "EEGProject", "channel_names": s_chans, "projector": projector,
               "feature_dim": proj_dim, "time_window": [0, train.n_samples],
               "init_temperature": tcfg.get("init_temperature", 0.07), "softplus": tcfg.get("softplus", True),
               "student_of": a.teacher, "condition": a.condition, "seed": a.seed, "curve": curve}
        save_run(student, os.path.join(a.runs_dir, a.save_as), cfg)
        print(f"saved student to {os.path.join(a.runs_dir, a.save_as)}")


if __name__ == "__main__":
    main()
