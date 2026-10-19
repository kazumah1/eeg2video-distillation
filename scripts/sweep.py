"""Run the distillation experiment grid and summarize it.

    python scripts/sweep.py run --groups channels losses --seeds 0 1 2 --results_csv results/sweep.csv
    python scripts/sweep.py summary --results_csv results/sweep.csv
    python scripts/sweep.py run --groups all --dry_run            # list the runs only

Resumable: a run whose tag is already in the CSV is skipped, so re-running after a
disconnect continues where it stopped. Every run is tagged "<config>|seed<n>"; the
summary averages over seeds and reports each config's paired gain over its baseline
(same seed, same montage, no distillation).

Groups (each answers one question):
  channels      Where does KD help? none vs response at 63/32/16/8 channels, plus an
                occipital-8 reference (same count as the Crown, best sites for vision).
  losses        Which signal transfers? emb, kl, feature, relational, all at crown8.
  teachers      Does privileged information matter? 17-ch teacher (higher accuracy,
                almost no Crown channels) and a born-again 8-ch teacher (same channels:
                pure regularization). Weight inheritance from the teacher, with/without KD.
  progressive   Teacher-assistant chain 63 -> 32 -> 16 -> 8, with and without weight
                inheritance at each step (progressive channel removal).
  single_trial  Realistic Crown use: train and test on single trials, no averaging.
"""
import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ["channels", "losses", "teachers", "progressive", "single_trial"]
METRICS = ["top1", "top5", "pct_teacher_retained", "fid_cos", "teacher_top1"]


def plan(a):
    """List of (config, seed, extra_args, save_as, baseline_config); order respects dependencies.

    The 8-channel no-KD student ("channels/crown8-none") is shared: it is the baseline for
    every crown8 comparison and is saved so it can serve as the born-again teacher.
    """
    T_ALL, T_P17 = a.teacher_all, a.teacher_post17
    runs, seen = [], set()

    def add(config, seed, args, save_as=None, base=None):
        if (config, seed) not in seen:
            seen.add((config, seed))
            runs.append((config, seed, args, save_as, base))

    def crown_base(seed):
        add("channels/crown8-none", seed, ["--teacher", T_ALL, "--montage", "crown8", "--condition", "none"],
            save_as=f"sweep_ba8_s{seed}")
        return "channels/crown8-none"

    groups = GROUPS if "all" in a.groups else a.groups
    for seed in a.seeds:
        if "channels" in groups:
            for m in ["all", "nested32", "nested16", "crown8", "occipital8"]:
                base = crown_base(seed) if m == "crown8" else f"channels/{m}-none"
                if m != "crown8":
                    add(base, seed, ["--teacher", T_ALL, "--montage", m, "--condition", "none"])
                add(f"channels/{m}-response", seed,
                    ["--teacher", T_ALL, "--montage", m, "--condition", "response"], base=base)
            # where does the Crown's signal come from? (no-KD only)
            for m in ["crown_po2", "crown_ant6"]:
                add(f"channels/{m}-none", seed, ["--teacher", T_ALL, "--montage", m, "--condition", "none"])
        if "losses" in groups:
            base = crown_base(seed)
            for c in ["emb", "kl", "response", "feature", "relational", "all"]:
                cfg = "channels/crown8-response" if c == "response" else f"losses/crown8-{c}"
                add(cfg, seed, ["--teacher", T_ALL, "--montage", "crown8", "--condition", c], base=base)
            # relational loss is ~0.1 vs ~1 (emb) and ~5 (kl) at weight 1, so also test it scaled up
            add("losses/crown8-relational-w10", seed,
                ["--teacher", T_ALL, "--montage", "crown8", "--condition", "relational", "--w_rel", "10"], base=base)
        if "teachers" in groups:
            base = crown_base(seed)
            add("teachers/crown8-response-t17", seed,
                ["--teacher", T_P17, "--montage", "crown8", "--condition", "response"], base=base)
            add("teachers/crown8-response-t17-init", seed,
                ["--teacher", T_P17, "--montage", "crown8", "--condition", "response", "--init_from_teacher", "1"],
                base=base)
            add("teachers/crown8-response-bornagain", seed,
                ["--teacher", os.path.join(a.runs_dir, f"sweep_ba8_s{seed}"), "--montage", "crown8",
                 "--condition", "response"], base=base)
            for c in ["none", "response"]:
                add(f"teachers/crown8-{c}-init63", seed,
                    ["--teacher", T_ALL, "--montage", "crown8", "--condition", c, "--init_from_teacher", "1"],
                    base=base)
        if "progressive" in groups:
            base = crown_base(seed)
            for init in [0, 1]:
                prev = T_ALL
                for m in ["nested32", "nested16", "crown8"]:
                    save = f"sweep_takd_init{init}_{m}_s{seed}" if m != "crown8" else None
                    add(f"progressive/{m}-takd-init{init}", seed,
                        ["--teacher", prev, "--montage", m, "--condition", "response", "--init_from_teacher", str(init)],
                        save_as=save, base=base if m == "crown8" else None)
                    prev = os.path.join(a.runs_dir, save) if save else prev
        if "single_trial" in groups:
            st = ["--train_average", "0", "--test_average", "0"]
            add("single_trial/all-none", seed, ["--teacher", T_ALL, "--montage", "all", "--condition", "none"] + st)
            add("single_trial/crown8-none", seed, ["--teacher", T_ALL, "--montage", "crown8", "--condition", "none"] + st)
            add("single_trial/crown8-response", seed,
                ["--teacher", T_ALL, "--montage", "crown8", "--condition", "response"] + st,
                base="single_trial/crown8-none")
    return runs


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def cmd_run(a):
    runs = plan(a)
    done = {r.get("tag") for r in read_rows(a.results_csv)}
    todo = [r for r in runs if f"{r[0]}|seed{r[1]}" not in done
            or (r[3] and not os.path.exists(os.path.join(a.runs_dir, r[3], "checkpoint_last.pth")))]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to run")
    for i, (config, seed, args, save_as, _) in enumerate(todo, 1):
        tag = f"{config}|seed{seed}"
        cmd = [sys.executable, os.path.join(ROOT, "train_student.py"), *args, "--seed", str(seed),
               "--tag", tag, "--results_csv", a.results_csv, "--runs_dir", a.runs_dir,
               "--epochs", str(a.epochs), *a.extra]
        if save_as:
            cmd += ["--save_as", save_as]
        print(f"[{i}/{len(todo)}] {tag}", flush=True)
        if a.dry_run:
            print("   ", " ".join(cmd[1:]))
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-2000:], out.stderr[-3000:])
            raise SystemExit(f"FAILED: {tag}")
        last = [l for l in out.stdout.splitlines() if l.startswith("epoch")][-1:]
        print("   ", last[0] if last else "", flush=True)


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, sd


def cmd_summary(a):
    rows = [r for r in read_rows(a.results_csv) if "|seed" in (r.get("tag") or "")]
    if not rows:
        raise SystemExit(f"no sweep rows in {a.results_csv}")
    a.groups, a.seeds = ["all"], sorted({int(r["seed"]) for r in rows})
    base_of = {cfg: base for cfg, _, _, _, base in plan(a)}
    by = defaultdict(dict)  # config -> seed -> row
    for r in rows:
        cfg = r["tag"].split("|seed")[0]
        by[cfg][int(r["seed"])] = r  # last row wins if a run was repeated
    out = []
    for cfg in sorted(by, key=lambda c: (c.split("/")[0], c)):
        seeds = by[cfg]
        line = {"config": cfg, "n": len(seeds), "channels": next(iter(seeds.values()))["n_channels"]}
        for m in METRICS:
            mu, sd = mean_std([float(r[m]) for r in seeds.values() if r.get(m) not in (None, "", "nan")])
            line[m] = f"{mu:.1f} ± {sd:.1f}" if not math.isnan(sd) else f"{mu:.1f}"
        base = base_of.get(cfg)
        if base and base in by:
            d = [float(seeds[s]["top1"]) - float(by[base][s]["top1"]) for s in seeds if s in by[base]]
            mu, sd = mean_std(d)
            line["d_top1_vs_base"] = f"{mu:+.1f} ± {sd:.1f}" if len(d) > 1 else (f"{mu:+.1f}" if d else "")
            line["wins"] = f"{sum(x > 0 for x in d)}/{len(d)}"
        out.append(line)
    cols = ["config", "channels", "n"] + METRICS + ["d_top1_vs_base", "wins"]
    widths = {c: max(len(c), *(len(str(l.get(c, ""))) for l in out)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for l in out:
        print("  ".join(str(l.get(c, "")).ljust(widths[c]) for c in cols))
    print("\nd_top1_vs_base: paired top-1 gain over the no-KD baseline at the same seed (points); "
          "wins: seeds where the gain is > 0. One test image = 0.5 points.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summary"])
    p.add_argument("--groups", nargs="+", default=["channels"], choices=GROUPS + ["all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--teacher_all", default="runs/nb_s08_all")
    p.add_argument("--teacher_post17", default="runs/nb_s08_post17")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--results_csv", default="results/sweep.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="anything after --extra is passed to every train_student.py call")
    a = p.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    main()
