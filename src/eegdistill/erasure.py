"""Controlled interventions on frozen EEG embeddings with a fixed image gallery."""

import numpy as np


def _unit_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def _retrieval(queries, gallery):
    scores = queries @ gallery.T
    correct = scores.diagonal().copy()
    np.fill_diagonal(scores, -np.inf)
    margins = correct - scores.max(axis=1)
    return {"top1": float(np.mean(margins > 1e-12)), "margin": float(margins.mean())}


def _probe_targets(train_labels, test_labels, categorical):
    targets = {}
    for name, values in train_labels.items():
        if name in categorical:
            classes, train_idx = np.unique(values, return_inverse=True)
            test_idx = np.searchsorted(classes, test_labels[name])
            if (len(classes) < 2 or len(np.unique(test_idx)) < 2
                    or not np.isin(test_labels[name], classes).all()):
                raise ValueError(f"{name}: need shared category labels and at least two classes per split")
            targets[name] = (np.eye(len(classes))[train_idx], test_idx)
        else:
            train = np.asarray(values, dtype=float)
            test = np.asarray(test_labels[name], dtype=float)
            if not np.isfinite(train).all() or not np.isfinite(test).all():
                raise ValueError(f"{name}: descriptors must be finite")
            # A 2-D descriptor (n, k) is one joint label: one k-column target.
            train, test = train.reshape(len(train), -1), test.reshape(len(test), -1)
            if not train.std(axis=0).all() or not test.std(axis=0).all():
                raise ValueError(f"{name}: descriptors must vary in both splits")
            targets[name] = (train, test)
    return targets


def _probe_scores(train, test, targets, categorical, ridge):
    mean = train.mean(axis=0)
    train = train - mean
    # Fit fresh probes after each intervention, with one solve for all outcomes.
    gram = (train.T @ train) / len(train)
    gram.flat[::len(gram) + 1] += ridge
    labels = np.concatenate([values for values, _ in targets.values()], axis=1)
    weights = np.linalg.solve(gram, (train.T @ labels) / len(train))
    predictions = (test - mean) @ weights + labels.mean(axis=0)
    scores = {}
    start = 0
    for name, (values, truth) in targets.items():
        end = start + values.shape[1]
        pred = predictions[:, start:end]
        start = end
        if name in categorical:
            correct = pred.argmax(axis=1) == truth
            accuracy = [correct[truth == label].mean() for label in np.unique(truth)]
            scores[name] = {"balanced_accuracy": float(np.mean(accuracy))}
        else:
            # Mean R^2 over columns (one column for a scalar descriptor).
            residual = np.sum((truth - pred) ** 2, axis=0)
            total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
            scores[name] = {"r2": float(np.mean(1 - residual / total))}
    return scores


def _control_delta(centered, basis, size):
    projected = (centered @ basis) @ basis.T
    norm = np.linalg.norm(projected, axis=1, keepdims=True)
    # A zero projection still needs a direction in the same control subspace.
    projected[norm[:, 0] == 0] = basis[:, 0]
    return _unit_rows(projected) * size[:, None]


def evaluate_erasure(train, test, gallery, train_labels, test_labels, *,
                     categorical=(), controls=20, strength=1.0, ridge=1e-4, seed=0):
    """Fit on training images and score on distinct test images in gallery order.

    Erase the orthogonal span of training cross-covariance directions. Edit EEG
    outputs before cosine normalization. Controls use random subspaces of the
    same dimension and match the pre-normalization L2 displacement per image.
    """
    train, test, gallery = [np.asarray(x, dtype=np.float64) for x in (train, test, gallery)]
    if (any(x.ndim != 2 or not np.isfinite(x).all() for x in (train, test, gallery))
            or test.shape != gallery.shape or train.shape[1] != test.shape[1]
            or len(train) < 2 or len(test) < 2):
        raise ValueError("expected finite train/test embeddings and one gallery row per test image")
    if (not train_labels or train_labels.keys() != test_labels.keys()
            or not set(categorical).issubset(train_labels)
            or any(np.shape(train_labels[k])[:1] != (len(train),)
                   or np.shape(test_labels[k])[:1] != (len(test),)
                   or np.ndim(train_labels[k]) > 2 - (k in categorical)
                   or np.shape(train_labels[k])[1:] != np.shape(test_labels[k])[1:]
                   for k in train_labels)):
        raise ValueError("expected matching label sets and one value (or row) per image")
    if controls < 2 or not 0 < strength <= 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("need at least two controls, strength in (0, 1], and positive finite ridge")
    if np.any(np.linalg.norm(gallery, axis=1) == 0):
        raise ValueError("gallery embeddings must have nonzero norms")

    targets = _probe_targets(train_labels, test_labels, categorical)
    gallery = _unit_rows(gallery)
    unit_test = _unit_rows(test)
    center = train.mean(axis=0)
    centered_train = train - center
    centered_test = test - center

    def score(edited_train, edited_test):
        queries = _unit_rows(edited_test)
        return {
            **_retrieval(queries, gallery),
            "probes": _probe_scores(_unit_rows(edited_train), queries, targets, categorical, ridge),
        }

    def score_edit(train_delta, test_delta, target_size):
        edited = test - test_delta
        realized_size = np.linalg.norm(test - edited, axis=1)
        unit_edited = _unit_rows(edited)
        valid_angles = (np.linalg.norm(test, axis=1) > 0) & (np.linalg.norm(edited, axis=1) > 0)
        cosine = np.sum(unit_test[valid_angles] * unit_edited[valid_angles], axis=1)
        angles = np.degrees(np.arccos(cosine.clip(-1, 1)))
        return {
            **score(train - train_delta, edited),
            "mean_l2": float(realized_size.mean()),
            "max_l2_matching_error": float(np.max(np.abs(realized_size - target_size))),
            "mean_angle_degrees": float(angles.mean()) if len(angles) else None,
            "angle_degrees_p05_p50_p95": (
                np.percentile(angles, [5, 50, 95]).tolist() if len(angles) else None
            ),
            "undefined_angles": int(len(test) - len(angles)),
        }

    result = {"baseline": score(train, test), "interventions": {}}
    for name, (labels, _) in targets.items():
        cross_cov = centered_train.T @ (labels - labels.mean(axis=0)) / len(train)
        left, singular, _ = np.linalg.svd(cross_cov, full_matrices=False)
        scale = np.linalg.norm(centered_train) * np.linalg.norm(labels - labels.mean(axis=0)) / len(train)
        tolerance = np.finfo(float).eps * max(cross_cov.shape) * scale
        rank = int((singular > tolerance).sum())
        if rank == 0 or rank == train.shape[1]:
            result["interventions"][name] = {"rank": rank, "status": "no_nontrivial_subspace"}
            continue
        basis = left[:, :rank]
        train_delta = strength * ((centered_train @ basis) @ basis.T)
        test_delta = strength * ((centered_test @ basis) @ basis.T)
        train_size = np.linalg.norm(train_delta, axis=1)
        test_size = np.linalg.norm(test_delta, axis=1)
        targeted = score_edit(train_delta, test_delta, test_size)
        draws = []
        rng = np.random.default_rng(seed)
        for _ in range(controls):
            random_basis, _ = np.linalg.qr(rng.normal(size=(train.shape[1], rank)))
            draws.append(score_edit(
                _control_delta(centered_train, random_basis, train_size),
                _control_delta(centered_test, random_basis, test_size),
                test_size,
            ))
        result["interventions"][name] = {
            "rank": rank, "target": targeted, "controls": draws,
            "target_l2_by_test_index": test_size.tolist(),
            "excess_damage": {
                metric: float(np.mean([draw[metric] for draw in draws]) - targeted[metric])
                for metric in ("top1", "margin")
            },
        }
    return result
