"""Decision-table verdicts for mi_erasure.py reports.

Participants are the unit of inference: control draws and seeds never add to n.
"""

import numpy as np


def subject_stats(values, seed=0, n_boot=10_000):
    """Mean and percentile bootstrap CI over subjects for {subject: value}."""
    x = np.asarray(list(values.values()), dtype=float)
    ci = None
    if len(x) > 1:
        rng = np.random.default_rng(seed)
        boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        ci = np.percentile(boot, [2.5, 97.5]).tolist()
    return {"n_subjects": len(x), "mean": float(x.mean()) if len(x) else None,
            "ci95": ci, "by_subject": dict(values)}


def _rows(report, prop, budget):
    rows = {}
    for subject, sub in report["subjects"].items():
        view = sub["views"].get(budget)
        if view and "excess_damage" in view["interventions"].get(prop, {}):
            rows[subject] = view
    return rows


def _probe(scores, name):
    (value,) = scores["probes"][name].values()
    return value


def _remaining(rows, edits, name, probe_chance, which="target"):
    """Mean over subjects of (edited - chance) / (baseline - chance) for probe `name`."""
    chance = probe_chance.get(name)
    if chance is None:
        metric = next(iter(next(iter(rows.values()))["baseline"]["probes"][name]))
        if metric != "r2":
            raise ValueError(f"{name}: pass probe_chance[{name!r}] (1 / n_classes)")
        chance = 0.0
    fractions = []
    for s, view in rows.items():
        base = _probe(view["baseline"], name) - chance
        if which == "target":
            edited = _probe(edits[s]["target"], name)
        else:
            edited = np.mean([_probe(c, name) for c in edits[s]["controls"]])
        fractions.append((edited - chance) / base if base > 0 else np.nan)
    return float(np.nanmean(fractions)) if not np.all(np.isnan(fractions)) else None


def classify(report, prop, budget, *, sesoi, floor_margin=0.01, removal_tol=0.1,
             collateral_tol=0.25, probe_chance=None, seed=0):
    """Map one property x repetition budget onto the decision table.

    floor_margin: top-1 above chance (1 / gallery size) still treated as the floor.
    removal_tol: largest fraction of above-chance held-out probe score the target
        erasure may leave before removal counts as incomplete.
    collateral_tol: largest extra fraction of a competing property's above-chance
        probe score the target edit may remove beyond the matched controls.
    probe_chance: {categorical property: 1 / n_classes}; r2 probes default to 0.
    """
    probe_chance = probe_chance or {}
    rows = _rows(report, prop, budget)
    if not rows:
        raise ValueError(f"{prop} @ {budget}: no subject has a nontrivial erasure")
    edits ={s: v["interventions"][prop] for s, v in rows.items()}
    chance = 1 / len(next(iter(edits.values()))["target_l2_by_test_index"])
    floor = chance + floor_margin
    baseline = np.mean([v["baseline"]["top1"] for v in rows.values()])
    target = np.mean([e["target"]["top1"] for e in edits.values()])
    control = np.mean([np.mean([c["top1"] for c in e["controls"]]) for e in edits.values()])
    excess = subject_stats({s: e["excess_damage"]["top1"] for s, e in edits.items()}, seed)
    remaining = _remaining(rows, edits, prop, probe_chance)
    # Derived targets (residualized, rank-matched) share a family with their source
    # property: losing the source is expected, not collateral.
    derived = report.get("derived") or {}
    root = lambda p: derived.get(p, {}).get("from", p)  # noqa: E731
    # Shuffled copies are meaningless labels: not targets of collateral concern.
    others = [p for p in next(iter(rows.values()))["baseline"]["probes"]
              if p != prop and derived.get(p, {}).get("kind") != "shuffled"]
    # A joint target's family also includes each of its member descriptors.
    family = {root(prop), *derived.get(prop, {}).get("of", [])}
    related = {p: _remaining(rows, edits, p, probe_chance)
               for p in others if root(p) in family}
    # Extra fraction of each competing property lost to the target edit beyond controls.
    collateral = {}
    for other in (p for p in others if p not in related):
        kept_target = _remaining(rows, edits, other, probe_chance, "target")
        kept_control = _remaining(rows, edits, other, probe_chance, "control")
        if kept_target is not None and kept_control is not None:
            collateral[other] = kept_control - kept_target

    # Prefer concept-shuffled copies of this property as the specificity baseline:
    # random subspaces understate the damage any data-fitted erasure does.
    shuffled = [p for p, d in derived.items() if d.get("kind") == "shuffled" and d["from"] == prop]
    vs_shuffled = None
    if shuffled:
        vs_shuffled = subject_stats({
            s: e["excess_damage"]["top1"] - np.mean([
                report["subjects"][s]["views"][budget]["interventions"][p]["excess_damage"]["top1"]
                for p in shuffled])
            for s, e in edits.items()}, seed)
    decide = vs_shuffled or excess

    verdict = "inconclusive"
    if baseline <= floor or (target <= floor and control <= floor):
        verdict = "floor"
    elif remaining is None or remaining > removal_tol:
        verdict = "incomplete_removal"
    elif any(loss > collateral_tol for loss in collateral.values()):
        verdict = "collateral"
    elif decide["ci95"] and decide["ci95"][0] > 0 and decide["mean"] >= sesoi:
        verdict = "specific_effect"
    elif decide["ci95"] and -sesoi < decide["ci95"][0] and decide["ci95"][1] < sesoi:
        verdict = "bounded_null"
    return {"verdict": verdict, "chance_top1": chance, "excess_top1": excess,
            "excess_vs_shuffled_top1": vs_shuffled,
            "target_remaining": remaining, "collateral_extra_loss": collateral,
            "related_remaining": related, "mean_top1": {"baseline": float(baseline), "target": float(target),
                          "control": float(control)}}


# Settings that must agree for a montage-by-intervention contrast to be paired.
SHARED_SETTINGS = ("annotations", "descriptors", "category", "strength", "gallery",
                   "controls", "ridge", "image_counts")


def montage_contrast(report_a, report_b, prop, budget, *, seed=0):
    """Paired per-subject difference in excess damage between two montages.

    D_a - D_b equals how much more the targeted edit shrinks the a-minus-b
    accuracy gap than matched random edits do. Compare this directly: an effect
    in one montage and none in the other does not show that the effects differ.
    """
    for key in SHARED_SETTINGS:
        if report_a.get(key) != report_b.get(key):
            raise ValueError(f"reports differ in {key}: {report_a.get(key)!r} vs {report_b.get(key)!r}")
    rows_a, rows_b = _rows(report_a, prop, budget), _rows(report_b, prop, budget)
    paired = sorted(rows_a.keys() & rows_b.keys(), key=int)
    unpaired = sorted(rows_a.keys() ^ rows_b.keys(), key=int)
    if len(paired) < 2:
        raise ValueError(f"need at least two paired subjects, got {len(paired)}")

    def excess(rows, s):
        return rows[s]["interventions"][prop]["excess_damage"]["top1"]

    return {
        "property": prop, "budget": budget, "paired_subjects": paired,
        "unpaired_subjects": unpaired,
        "excess_top1_a": subject_stats({s: excess(rows_a, s) for s in paired}, seed),
        "excess_top1_b": subject_stats({s: excess(rows_b, s) for s in paired}, seed),
        "excess_top1_a_minus_b": subject_stats(
            {s: excess(rows_a, s) - excess(rows_b, s) for s in paired}, seed),
        "baseline_gap_top1": subject_stats(
            {s: rows_a[s]["baseline"]["top1"] - rows_b[s]["baseline"]["top1"]
             for s in paired}, seed),
    }


def paired_contrast(report, prop_a, prop_b, budget, *, seed=0):
    """Per-subject excess damage of prop_a minus prop_b in one report.

    Use for the headline comparison (e.g. residualized edges vs rank-1 category):
    one significant and one non-significant effect does not show they differ.
    """
    rows_a, rows_b = _rows(report, prop_a, budget), _rows(report, prop_b, budget)
    paired = sorted(rows_a.keys() & rows_b.keys(), key=int)
    if len(paired) < 2:
        raise ValueError(f"need at least two subjects with both {prop_a} and {prop_b}")

    def excess(rows, prop, s):
        return rows[s]["interventions"][prop]["excess_damage"]["top1"]

    return {
        "a": prop_a, "b": prop_b, "budget": budget, "paired_subjects": paired,
        "excess_top1_a_minus_b": subject_stats(
            {s: excess(rows_a, prop_a, s) - excess(rows_b, prop_b, s) for s in paired}, seed),
    }


# Settings that must agree before per-subject runner outputs pool into one report.
MERGE_SETTINGS = SHARED_SETTINGS + ("derived", "seed", "eeg_normalization", "method")


def merge_reports(reports):
    """Pool per-subject mi_erasure reports (one runner call each) into one report.

    Settings must match exactly; each subject may appear once. The per-report
    summaries are dropped: they cover one subject each and do not pool.
    """
    first, *rest = reports
    merged = {k: v for k, v in first.items() if k not in ("subjects", "summary")}
    merged["subjects"] = dict(first["subjects"])
    for other in rest:
        for key in MERGE_SETTINGS:
            if other.get(key) != first.get(key):
                raise ValueError(f"reports differ in {key}: {first.get(key)!r} vs {other.get(key)!r}")
        for subject, sub in other["subjects"].items():
            if subject in merged["subjects"]:
                raise ValueError(f"subject {subject} appears in more than one report")
            merged["subjects"][subject] = sub
    return merged
