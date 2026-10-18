"""EEG encoders and the model wrapper used for teachers and students.

`EEGProject`, `ResidualAdd` and the projectors are copied from NeuroBridge
(https://github.com/feroooooo/NeuroBridge, MIT License, (c) 2025 Wenjiang Zhang)
with identical module names, so NeuroBridge checkpoints load directly.
"""
import torch
import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGProject(nn.Module):
    def __init__(self, feature_dim=1024, eeg_sample_points=250, channels_num=63):
        super().__init__()
        self.input_dim = eeg_sample_points * channels_num
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Dropout(0.3),
            )),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x, return_hidden=False):
        x = x.reshape(x.shape[0], self.input_dim)
        h = self.model[0](x)          # first layer: the only one that sees channels
        out = self.model[2](self.model[1](h))
        return (out, h) if return_hidden else out


class ProjectorLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class ProjectorDirect(nn.Module):
    def forward(self, x):
        return x


def make_projector(kind, in_dim, out_dim):
    if kind == "direct":
        return ProjectorDirect()
    if kind == "linear":
        return ProjectorLinear(in_dim, out_dim)
    raise ValueError(f"unsupported projector '{kind}'")


class EEGImageModel(nn.Module):
    """Encoder + EEG projector + image projector, as in NeuroBridge.

    `channel_idx` are indices into the dataset's full channel list; the model
    selects them itself, so teacher and student can share one batch.
    """

    def __init__(self, channel_idx, n_samples, feat_dim, projector="linear", proj_dim=512):
        super().__init__()
        self.register_buffer("channel_idx", torch.as_tensor(channel_idx, dtype=torch.long))
        self.encoder = EEGProject(feature_dim=feat_dim, eeg_sample_points=n_samples,
                                  channels_num=len(channel_idx))
        self.eeg_projector = make_projector(projector, feat_dim, proj_dim)
        self.img_projector = make_projector(projector, feat_dim, proj_dim)

    def encode_eeg(self, x_full, return_hidden=False):
        x = x_full.index_select(1, self.channel_idx)
        out, h = self.encoder(x, return_hidden=True)
        z = self.eeg_projector(out)
        return (z, h) if return_hidden else z

    def encode_img(self, img_feat):
        return self.img_projector(img_feat)
