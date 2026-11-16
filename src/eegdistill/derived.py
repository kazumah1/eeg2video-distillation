"""Derived erasure targets that evaluate_erasure can consume as ordinary descriptors.

residualized: a descriptor minus its train-category mean, so its erased direction
    carries no linear category signal by construction on train images.
rank-matched category: a scalar whose train cross-covariance with the embeddings
    spans only the top category direction, so a category edit has the same rank
    (1) as a scalar descriptor edit.
"""

import numpy as np


def add_derived_labels(train_z, train_labels, test_labels, *, category,
                       residualize=(), rank_matched=True, shuffles=0, train_groups=None,
                       seed=0):
    """Return (train_labels, test_labels, derived) with derived columns added.

    shuffles: number of concept-shuffled copies of every original property.
        Train values move as whole concept blocks (train_groups gives each image's
        concept; concepts must be equal-sized), test values are permuted across
        images. A shuffled label keeps the data-fitted, concept-level structure of
        an erasure but not its meaning: erasing it estimates the generic damage of
        removing concept-identity directions, which random subspaces understate.
    """
    train_labels, test_labels = dict(train_labels), dict(test_labels)
    originals = list(train_labels)
    derived = {}
    train_cat = np.asarray(train_labels[category])
    test_cat = np.asarray(test_labels[category])
    if not np.isin(test_cat, train_cat).all():
        raise ValueError(f"{category}: test categories missing from train")
    for name in residualize:
        values = np.asarray(train_labels[name], dtype=float)
        means = {c: values[train_cat == c].mean() for c in np.unique(train_cat)}
        key = f"{name}_resid_{category}"
        train_labels[key] = values - np.array([means[c] for c in train_cat])
        test_labels[key] = (np.asarray(test_labels[name], dtype=float)
                            - np.array([means[c] for c in test_cat]))
        derived[key] = {"kind": "residualized", "from": name, "on": category}
    if rank_matched:
        # Same raw (pre-normalization) embeddings evaluate_erasure edits.
        z = np.asarray(train_z, dtype=float)
        classes = np.unique(train_cat)
        onehot = (train_cat[:, None] == classes).astype(float)
        mean = onehot.mean(axis=0)
        cross_cov = (z - z.mean(axis=0)).T @ (onehot - mean) / len(z)
        _, _, vt = np.linalg.svd(cross_cov, full_matrices=False)
        top = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])  # deterministic sign
        key = f"{category}_rank1"
        train_labels[key] = (onehot - mean) @ top
        test_labels[key] = ((test_cat[:, None] == classes).astype(float) - mean) @ top
        derived[key] = {"kind": "rank_matched", "from": category, "rank": 1}
    if shuffles:
        groups = np.asarray(train_groups)
        names, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
        if len(groups) != len(train_cat) or len(set(sizes)) != 1:
            raise ValueError("shuffles need one equal-sized concept group per train image")
        # Row positions of each concept's images, in their original order.
        members = [np.flatnonzero(inverse == g) for g in range(len(names))]
        rng = np.random.default_rng(seed)
        for k in range(shuffles):
            for name in originals:
                values = np.asarray(train_labels[name])
                moved = values.copy()
                for dest, src in zip(members, rng.permutation(len(members))):
                    moved[dest] = values[members[src]]
                key = f"{name}_shuf{k}"
                train_labels[key] = moved
                test_labels[key] = rng.permutation(np.asarray(test_labels[name]))
                derived[key] = {"kind": "shuffled", "from": name}
    return train_labels, test_labels, derived
