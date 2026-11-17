"""Run controlled EEG embedding erasure on saved NeuroBridge-format checkpoints."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eegdistill.checkpoints import load_run, teacher_channel_names
from eegdistill.data import ThingsEEG, load_ch_names
from eegdistill.derived import add_derived_labels
from eegdistill.erasure import evaluate_erasure
from eegdistill.evaluate import embed


def load_annotations(path, sizes, descriptors, category):
    columns = descriptors + ([category] if category else [])
    required = {"split", "index", "image_id", *columns}
    if len(required) != len(columns) + 3:
        raise ValueError("annotation columns must be distinct")
    rows = {split: {} for split in sizes}
    seen_ids = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"annotation CSV requires {', '.join(sorted(required))}")
        for row in reader:
            split = row["split"]
            index = int(row["index"])
            image_id = (row["image_id"] or "").strip()
            if split not in rows or not 0 <= index < sizes[split] or index in rows[split]:
                raise ValueError(f"invalid or repeated annotation index: {split}/{index}")
            if not image_id or image_id in seen_ids:
                raise ValueError(f"empty or duplicate image_id at {split}/{index}: {image_id!r}")
            seen_ids.add(image_id)
            rows[split][index] = {column: (row[column] or "").strip() for column in columns}
    labels = {}
    for split, size in sizes.items():
        if len(rows[split]) != size:
            raise ValueError(f"{split}: annotations cover {len(rows[split])} of {size} images")
        labels[split] = {}
        for column in columns:
            values = [rows[split][index][column] for index in range(size)]
            if "" in values:
                raise ValueError(f"{split}: missing value in {column}")
            labels[split][column] = np.asarray(values, dtype=str if column == category else float)
    return labels


def run_subject(args, subject, run_dir, labels, eeg_info):
    config = json.loads((run_dir / "train_config.json").read_text())
    eeg_file = args.eeg_dir / f"sub-{subject:02d}" / "test.npy"
    shape = np.load(eeg_file, mmap_mode="r").shape
    start, end = config.get("time_window", [0, shape[-1]])
    if not 0 <= start < end <= shape[-1]:
        raise ValueError(f"{run_dir}: invalid checkpoint time window {start, end}")
    if config.get("eeg_test_aug"):
        raise ValueError(f"{run_dir}: checkpoint evaluation uses EEG augmentation")
    if config.get("image_test_aug") and args.eval_img == "clean":
        raise ValueError(f"{run_dir}: checkpoint expects an augmented test gallery")
    expected_test = config.get("test_subject_ids")
    if expected_test and subject not in expected_test:
        raise ValueError(f"{run_dir}: subject {subject} absent from checkpoint's test subjects")

    budgets = list(dict.fromkeys(shape[2] if n == "all" else int(n) for n in args.repeats))
    if any(not 1 <= n <= shape[2] for n in budgets):
        raise ValueError(f"subject {subject}: repetition budget must be in [1, {shape[2]}]")
    common = {
        "eeg_dir": args.eeg_dir, "feat_dir": args.feat_dir, "subjects": [subject],
        "time_window": (start, end), "channels": teacher_channel_names(run_dir),
    }
    train = ThingsEEG(**common, split="train")
    model, _, channels = load_run(run_dir, train.ch_names, train.n_samples, train.feat_dim,
                                  device=args.device)
    train_z, _ = embed(model, train, args.device, args.batch_size)
    del train
    report = {
        "run_dir": str(run_dir.resolve()), "checkpoint": "checkpoint_last.pth",
        "channels": channels, "time_window": [start, end], "views": {},
        "input_is_subset_of_mvnn_data": (
            eeg_info.get("normalization") == "mvnn" and len(channels) < len(eeg_info["ch_names"])
        ),
    }
    train_labels, test_labels, derived = labels["train"], labels["test"], {}
    if args.residualize or args.rank_matched_category or args.shuffled_controls or args.joint:
        # Rank-matched category depends on this subject's embeddings, so derive per subject.
        # Train images are stored (object, image)-flattened, so index // n_img is the concept.
        n_img = np.load(args.feat_dir / "image_train.npy", mmap_mode="r").shape[1]
        train_labels, test_labels, derived = add_derived_labels(
            train_z, train_labels, test_labels, category=args.category,
            residualize=args.residualize, rank_matched=args.rank_matched_category,
            joint=args.joint, shuffles=args.shuffled_controls,
            train_groups=np.arange(len(train_z)) // n_img, seed=args.seed)
    report["derived"] = derived
    for n in budgets:
        test = ThingsEEG(**common, split="test", repeats=n, aug_dir=args.aug_dir,
                         eval_img=args.eval_img)
        test_z, gallery = embed(model, test, args.device, args.batch_size)
        report["views"][str(n)] = evaluate_erasure(
            train_z, test_z, gallery, train_labels, test_labels,
            categorical=tuple(name for name in train_labels if name == args.category
                              or derived.get(name, {}) == {"kind": "shuffled",
                                                           "from": args.category}), controls=args.controls,
            strength=args.strength, ridge=args.ridge, seed=args.seed,
        )
    return report


def subject_stats(values, seed):
    subjects = list(values)
    x = np.asarray(list(values.values()))
    if not len(x):
        return {"n_subjects": 0, "mean": None, "bootstrap_95_ci": None, "by_subject": values}
    ci = None
    if len(x) > 1:
        rng = np.random.default_rng(seed)
        sample = rng.choice(x, size=(10_000, len(x)), replace=True).mean(axis=1)
        ci = np.percentile(sample, [2.5, 97.5]).tolist()
    return {"n_subjects": len(subjects), "mean": float(x.mean()),
            "bootstrap_95_ci": ci, "by_subject": values}


def summarize(subjects, descriptors, category, seed):
    summary = {}
    budgets = sorted({int(n) for subject in subjects.values() for n in subject["views"]})
    for budget in map(str, budgets):
        interventions = {
            key: sub["views"][budget]["interventions"]
            for key, sub in subjects.items() if budget in sub["views"]
        }
        names = descriptors + ([category] if category else [])
        effects = {
            name: {key: row[name]["excess_damage"] for key, row in interventions.items()
                   if "excess_damage" in row[name]}
            for name in names
        }
        by_property = {
            name: {metric: subject_stats({s: row[metric] for s, row in data.items()}, seed)
                   for metric in ("top1", "margin")}
            for name, data in effects.items()
        }
        summary[budget] = {"effects": by_property}
        if category:
            summary[budget]["paired_contrasts"] = {
                f"{name}_minus_{category}": {
                    metric: subject_stats({
                        s: row[metric] - effects[category][s][metric]
                        for s, row in effects[name].items() if s in effects[category]
                    }, seed)
                    for metric in ("top1", "margin")
                }
                for name in descriptors
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("SUBJECT", "RUN_DIR"),
                        required=True, help="subject ID and frozen run directory, repeatable")
    parser.add_argument("--eeg-dir", type=Path, required=True)
    parser.add_argument("--feat-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--descriptor", action="append", default=[], metavar="COLUMN")
    parser.add_argument("--category", metavar="COLUMN")
    parser.add_argument("--residualize", action="append", default=[], metavar="COLUMN",
                        help="also erase this descriptor residualized on train-category means")
    parser.add_argument("--shuffled-controls", type=int, default=0, metavar="N",
                        help="also erase N concept-shuffled copies of every property")
    parser.add_argument("--rank-matched-category", action="store_true",
                        help="also erase only the top category direction (rank 1)")
    parser.add_argument("--joint", action="append", default=[], metavar="NAME=COL,COL",
                        help="also erase these descriptors together as one multi-column target")
    parser.add_argument("--aug-dir", type=Path)
    parser.add_argument("--eval-img", choices=("clean", "augmean", "augrand"), default="clean")
    parser.add_argument("--repeats", nargs="+", default=["all"], metavar="N_OR_ALL")
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if not (args.descriptor or args.category):
        parser.error("provide at least one --descriptor or --category column")
    if (args.aug_dir is None) != (args.eval_img == "clean"):
        parser.error("augmented gallery modes require --aug-dir")
    try:
        args.joint = {name: cols.split(",") for name, cols in
                      (item.split("=", 1) for item in args.joint)}
    except ValueError:
        parser.error("--joint takes NAME=COL,COL")
    if any(len(cols) < 2 or not set(cols) <= set(args.descriptor) for cols in args.joint.values()):
        parser.error("--joint needs two or more --descriptor columns")
    if (args.residualize or args.rank_matched_category or args.shuffled_controls
            or args.joint) and not args.category:
        parser.error("--residualize, --rank-matched-category, --shuffled-controls and "
                     "--joint need --category")
    if not set(args.residualize) <= set(args.descriptor):
        parser.error("--residualize columns must also be --descriptor columns")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        runs = [(int(subject), Path(run_dir)) for subject, run_dir in args.run]
        if len({subject for subject, _ in runs}) != len(runs):
            raise ValueError("provide one run per subject")
        sizes = {
            split: int(np.prod(np.load(args.feat_dir / f"image_{split}.npy",
                                       mmap_mode="r").shape[:2]))
            for split in ("train", "test")
        }
        labels = load_annotations(args.annotations, sizes, args.descriptor, args.category)
        eeg_info = json.loads((args.eeg_dir / "info.json").read_text())
        subjects = {subject: run_subject(args, subject, run_dir, labels, eeg_info)
                    for subject, run_dir in runs}
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    derived = next(iter(subjects.values()))["derived"]
    descriptors = args.descriptor + list(derived)
    report = {
        "intervention": "EEG projector output before cosine normalization; fixed image gallery",
        "method": "orthogonal projection of training cross-covariance directions",
        "annotations": str(args.annotations.resolve()), "descriptors": descriptors,
        "derived": derived,
        "category": args.category, "eeg_dir": str(args.eeg_dir.resolve()),
        "eeg_normalization": eeg_info.get("normalization"), "image_counts": sizes,
        "feat_dir": str(args.feat_dir.resolve()), "gallery": args.eval_img,
        "aug_dir": str(args.aug_dir.resolve()) if args.aug_dir else None,
        "controls": args.controls, "strength": args.strength, "ridge": args.ridge,
        "seed": args.seed, "tie_policy": "a tie for top score counts as a miss",
        "repetition_policy": "prefix of stored EEG repetitions; all train repetitions",
        "subjects": subjects, "summary": summarize(subjects, descriptors, args.category, args.seed),
    }
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
