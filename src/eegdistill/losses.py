"""Task loss and the three distillation losses.

The task loss mirrors NeuroBridge's intra-subject config (--img_l2norm,
--softplus, no --eeg_l2norm). Distillation losses compare normalized
embeddings; test-time retrieval uses cosine similarity, as NeuroBridge does.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE between EEG and image embeddings in a batch.

    NeuroBridge's `--softplus` setting uses scale = softplus(log(1/T)); we keep
    that so our students train under the same objective as the teacher.
    """

    def __init__(self, init_temperature=0.07, softplus=True, eeg_l2norm=False, img_l2norm=True):
        super().__init__()
        self.register_buffer("log_inv_t", torch.tensor(math.log(1 / init_temperature)))
        self.softplus = softplus
        self.eeg_l2norm, self.img_l2norm = eeg_l2norm, img_l2norm

    def scale(self):
        return F.softplus(self.log_inv_t) if self.softplus else self.log_inv_t.exp()

    def logits(self, z_eeg, z_img):
        e = F.normalize(z_eeg, dim=1) if self.eeg_l2norm else z_eeg
        i = F.normalize(z_img, dim=1) if self.img_l2norm else z_img
        return e @ i.T * self.scale()

    def forward(self, z_eeg, z_img):
        logits = self.logits(z_eeg, z_img)
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def response_embedding_loss(z_s, z_t):
    """1 - cosine between student and teacher EEG embeddings."""
    return (1 - F.cosine_similarity(z_s, z_t.detach(), dim=1)).mean()


def response_kl_loss(z_s, z_t, z_img, tau=4.0):
    """Hinton KD on retrieval scores: match the teacher's softened distribution
    over the batch's images. Scaled by tau^2 as in Hinton et al. (2015)."""
    img = F.normalize(z_img, dim=1)
    s = F.normalize(z_s, dim=1) @ img.T
    t = F.normalize(z_t.detach(), dim=1) @ img.T
    # cosine in [-1, 1]; multiply by 1/0.07 so tau acts on the usual logit scale
    s, t = s / 0.07, t / 0.07
    return F.kl_div(F.log_softmax(s / tau, dim=1), F.softmax(t / tau, dim=1),
                    reduction="batchmean") * tau ** 2


def feature_loss(h_s, h_t):
    """1 - cosine between hidden layers. Both are the encoder's first-layer
    output (same width), which is after spatial mixing, so 63 vs 8 inputs
    have a comparable representation here."""
    return (1 - F.cosine_similarity(h_s, h_t.detach(), dim=1)).mean()


def relational_loss(z_s, z_t):
    """Similarity-preserving KD: match the trial-by-trial cosine matrix."""
    zs, zt = F.normalize(z_s, dim=1), F.normalize(z_t.detach(), dim=1)
    return F.mse_loss(zs @ zs.T, zt @ zt.T)
