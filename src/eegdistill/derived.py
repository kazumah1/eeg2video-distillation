"""Derived erasure targets that evaluate_erasure can consume as ordinary descriptors.

residualized: a descriptor minus its train-category mean, so its erased direction
    carries no linear category signal by construction on train images.
rank-matched category: a scalar whose train cross-covariance with the embeddings
    spans only the top category direction, so a category edit has the same rank
    (1) as a scalar descriptor edit.
"""

import numpy as np


def add_derived_labels(train_z, train_labels, test_labels, *, category,
                       residualize=(), rank_matched=True):
    """Return (train_labels, test_labels, derived) with derived columns added."""
    train_labels, test_labels = dict(train_labels), dict(test_labels)
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
    return train_labels, test_labels, derived
