"""Decision-table verdicts (and optional montage contrast) for mi_erasure.py reports.

  python mi_verdict.py occ8.json --sesoi 0.02 --probe-chance coarse_category=0.1
  python mi_verdict.py occ8.json --contrast edge_density_resid_category category_rank1 \
      --sesoi 0.02 --probe-chance coarse_category=0.1
  python mi_verdict.py occ8/sub-*.json --vs crown8/sub-*.json --sesoi 0.02 --probe-chance coarse_category=0.1
  python mi_verdict.py results/erasure/sub-*.json --sesoi 0.02 --probe-chance coarse_category=0.1

Several report files (one runner call per subject) are pooled with merge_reports.

Freeze --sesoi and the tolerances before looking at the official run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from eegdistill.verdict import classify, merge_reports, montage_contrast, paired_contrast


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, nargs="+",
                        help="mi_erasure.py JSON(s) for montage a; several are pooled")
    parser.add_argument("--vs", type=Path, nargs="+",
                        help="mi_erasure.py JSON(s) for montage b; several are pooled")
    parser.add_argument("--sesoi", type=float, required=True,
                        help="smallest top-1 excess damage worth reporting (fraction)")
    parser.add_argument("--probe-chance", action="append", default=[], metavar="NAME=P",
                        help="chance balanced accuracy for each categorical property")
    parser.add_argument("--contrast", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="per-subject excess damage A minus B, e.g. "
                             "edge_density_resid_category category_rank1")
    parser.add_argument("--size-matched", action="store_true",
                        help="decide against shuffled copies rescaled to the target edit size")
    parser.add_argument("--floor-margin", type=float, default=0.01)
    parser.add_argument("--removal-tol", type=float, default=0.1)
    parser.add_argument("--collateral-tol", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        chance = {k: float(v) for k, v in (item.split("=", 1) for item in args.probe_chance)}
        report = merge_reports([json.loads(p.read_text()) for p in args.report])
        props = report["descriptors"] + ([report["category"]] if report.get("category") else [])
        props = [p for p in props
                 if (report.get("derived") or {}).get(p, {}).get("kind") != "shuffled"]
        budgets = sorted({b for s in report["subjects"].values() for b in s["views"]}, key=int)
        out = {"settings": vars(args) | {"report": [str(p) for p in args.report],
                                          "vs": [str(p) for p in args.vs] if args.vs else None},
               "verdicts": {}}
        for budget in budgets:
            out["verdicts"][budget] = {
                prop: classify(report, prop, budget, sesoi=args.sesoi,
                               floor_margin=args.floor_margin, removal_tol=args.removal_tol,
                               collateral_tol=args.collateral_tol, probe_chance=chance,
                               seed=args.seed, size_matched=args.size_matched)
                for prop in props
            }
        out["contrasts"] = {
            budget: {f"{a}_minus_{b}": paired_contrast(report, a, b, budget, seed=args.seed)
                     for a, b in args.contrast}
            for budget in budgets
        }
        if args.vs:
            other = merge_reports([json.loads(p.read_text()) for p in args.vs])
            out["montage_contrast"] = {
                budget: {prop: montage_contrast(report, other, prop, budget, seed=args.seed,
                                               size_matched=args.size_matched)
                         for prop in props}
                for budget in budgets
            }
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    json.dump(out, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
