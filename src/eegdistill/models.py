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


@torch.no_grad()
def init_from_teacher(student, teacher, student_chans, teacher_chans):
    """Weight inheritance: copy everything the student can reuse from the teacher.

    EEGProject flattens (C, T) row-major, so channel c owns input columns
    [c*T, (c+1)*T) of the first Linear. Columns of student channels the teacher
    also has are copied; the rest keep their random init. All later layers and
    both projectors are copied whole. Returns the number of channels inherited.
    """
    T = student.encoder.input_dim // len(student_chans)
    w_s, w_t = student.encoder.model[0].weight, teacher.encoder.model[0].weight
    shared = 0
    for i, c in enumerate(student_chans):
        if c in teacher_chans:
            j = teacher_chans.index(c)
            w_s[:, i * T:(i + 1) * T] = w_t[:, j * T:(j + 1) * T]
            shared += 1
    student.encoder.model[0].bias.copy_(teacher.encoder.model[0].bias)
    student.encoder.model[1].load_state_dict(teacher.encoder.model[1].state_dict())
    student.encoder.model[2].load_state_dict(teacher.encoder.model[2].state_dict())
    student.eeg_projector.load_state_dict(teacher.eeg_projector.state_dict())
    student.img_projector.load_state_dict(teacher.img_projector.state_dict())
    return shared
