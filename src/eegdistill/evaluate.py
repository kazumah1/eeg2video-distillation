"""Zero-shot retrieval and student-teacher fidelity."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed(model, dataset, device, batch_size=1024):
    model.eval()
    zs = []
    for i in range(0, len(dataset), batch_size):
        x = dataset.eeg[i:i + batch_size].to(device)
        zs.append(model.encode_eeg(x).float().cpu())
    z_img = model.encode_img(dataset.image_features.to(device)).float().cpu()
    return torch.cat(zs), z_img


def retrieval(z_eeg, z_img, target_idx, ks=(1, 5)):
    """Top-k accuracy (%) of each EEG embedding against all candidate images."""
    sim = F.normalize(z_eeg, dim=1) @ F.normalize(z_img, dim=1).T
    ranks = (sim > sim.gather(1, target_idx[:, None])).sum(1)  # 0 = correct is top-1
    return {f"top{k}": float((ranks < k).float().mean() * 100) for k in ks}


def chance_normalized(acc_pct, n_candidates, k=1):
    chance = 100.0 * k / n_candidates
    return (acc_pct - chance) / (100.0 - chance)


@torch.no_grad()
def fidelity(z_s, z_t, z_img_s, z_img_t):
    """Embedding cosine and top-1 agreement between student and teacher."""
    cos = F.cosine_similarity(z_s, z_t, dim=1).mean().item()
    pick_s = (F.normalize(z_s, dim=1) @ F.normalize(z_img_s, dim=1).T).argmax(1)
    pick_t = (F.normalize(z_t, dim=1) @ F.normalize(z_img_t, dim=1).T).argmax(1)
    return {"fid_cos": cos, "fid_top1_agree": float((pick_s == pick_t).float().mean() * 100)}


def evaluate(model, dataset, device, teacher=None):
    z, z_img = embed(model, dataset, device)
    res = retrieval(z, z_img, dataset.img_idx)
    n = z_img.shape[0]
    res["top1_chance_norm"] = chance_normalized(res["top1"], n, 1)
    res["n_candidates"] = n
    if teacher is not None:
        zt, zt_img = embed(teacher, dataset, device)
        t_res = retrieval(zt, zt_img, dataset.img_idx)
        res["teacher_top1"] = t_res["top1"]
        res["teacher_top5"] = t_res["top5"]
        t_cn = chance_normalized(t_res["top1"], n, 1)
        res["pct_teacher_retained"] = 100 * res["top1_chance_norm"] / t_cn if t_cn > 0 else float("nan")
        res.update(fidelity(z, zt, z_img, zt_img))
    return res
