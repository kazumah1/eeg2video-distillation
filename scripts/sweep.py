"""Run the distillation experiment grid and summarize it.

    python scripts/sweep.py run --groups core --subject 3 --seeds 0 1 2 --results_csv results/subjects.csv
    python scripts/sweep.py run --groups pooled --results_csv results/subjects.csv
    python scripts/sweep.py run --groups loso --subject 3 --seeds 0 --results_csv results/subjects.csv
    python scripts/sweep.py summary --results_csv results/subjects.csv [--subjects 1 2 3]
    python scripts/sweep.py run --groups all --dry_run            # list the runs only

Resumable: a run whose (subject, tag) is already in the CSV is skipped, so re-running after a
disconnect continues where it stopped. Runs are tagged "<config>|seed<n>". The summary pools
all (subject, seed) pairs per config and reports the paired top-1 gain over the config's
baseline, matched on subject and seed.

Teacher paths are templates filled with the subject: --teacher_all "runs/nb_s{subject:02d}_all".

Groups (each answers one question):
  channels      Where does KD help? none vs response at 63/32/16/8 channels, an occipital-8
                reference, and the Crown diagnostics (PO3/PO4 only, the other 6 sites).
  losses        Which signal transfers? emb, kl, feature, relational, all, imitate at crown8.
  teachers      Does privileged information matter? 17-ch teacher, born-again 8-ch teacher,
                weight inheritance.
  progressive   Teacher-assistant chain 63 -> 32 -> 16 -> 8, with/without weight inheritance.
  single_trial  Train and test on single trials, no averaging.
  agnostic      One channel-agnostic model evaluated on every montage.
  core          The per-subject replication set: the key rows of channels + teachers.
  pooled        Train one student on all subjects' data (--pool_subjects), test on each
                subject. Is the 8-ch student limited by data or by information?
  loso          Leave one subject out: train on the others, test on --subject; also with
                10% of the subject's own data as a short calibration set.
"""
import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = ["channels", "losses", "teachers", "progressive", "single_trial", "agnostic", "core", "pooled", "loso"]
METRICS = ["top1", "top5", "pct_teacher_retained", "fid_cos", "teacher_top1"]


def plan(a):
    """List of run dicts (config, seed, subject, args, post, save_as, base, script), in dependency order.

    `subject` is the subject whose CSV row marks the run as done. `post` args come after the
    shared ones (--epochs etc.) so they can override them.
    """
    S = a.subject
    T_ALL, T_P17 = a.teacher_all.format(subject=S), a.teacher_post17.format(subject=S)
    T_POOL, T_LOSO = a.teacher_pool.format(subject=S), a.teacher_loso.format(subject=S)
    pool = a.pool_subjects
    runs, seen = [], set()

    def add(config, seed, args, save_as=None, base=None, script="train_student.py", subject=S, post=()):
        if (config, seed, subject) not in seen:
            seen.add((config, seed, subject))
            runs.append(dict(config=config, seed=seed, subject=subject, args=args, post=list(post),
                             save_as=save_as, base=base, script=script))

    def ba_name(seed):
        return f"sweep_s{S:02d}_ba8_s{seed}"

    def crown_base(seed):  # shared crown8 no-KD baseline, saved as the born-again teacher
        add("channels/crown8-none", seed, ["--teacher", T_ALL, "--montage", "crown8", "--condition", "none"],
            save_as=ba_name(seed))
        return "channels/crown8-none"

    def none_run(m, seed, teacher=T_ALL):
        add(f"channels/{m}-none", seed, ["--teacher", teacher, "--montage", m, "--condition", "none"])
        return f"channels/{m}-none"

    groups = GROUPS if "all" in a.groups else a.groups
    for seed in a.seeds:
        if "channels" in groups or "core" in groups:
            full = "channels" in groups
            for m in (["all", "nested32", "nested16", "crown8", "occipital8"] if full
                      else ["all", "nested32", "crown8", "occipital8"]):
                base = crown_base(seed) if m == "crown8" else none_run(m, seed)
                if full or m in ("all", "crown8", "occipital8"):
                    add(f"channels/{m}-response", seed,
                        ["--teacher", T_ALL, "--montage", m, "--condition", "response"], base=base)
            for m in (["crown_po2", "crown_ant6"] if full else ["crown_po2"]):
                none_run(m, seed)
        if "losses" in groups:
            base = crown_base(seed)
            for c in ["emb", "kl", "response", "feature", "relational", "all", "imitate"]:
                cfg = "channels/crown8-response" if c == "response" else f"losses/crown8-{c}"
                add(cfg, seed, ["--teacher", T_ALL, "--montage", "crown8", "--condition", c], base=base)
            # relational loss is ~0.1 vs ~1 (emb) and ~5 (kl) at weight 1, so also test it scaled up
            add("losses/crown8-relational-w10", seed,
                ["--teacher", T_ALL, "--montage", "crown8", "--condition", "relational", "--w_rel", "10"], base=base)
        if "teachers" in groups or "core" in groups:
            base = crown_base(seed)
            add("teachers/crown8-response-t17", seed,
                ["--teacher", T_P17, "--montage", "crown8", "--condition", "response"], base=base)
            add("teachers/crown8-response-t17-init", seed,   # best Crown setup: keep the model
                ["--teacher", T_P17, "--montage", "crown8", "--condition", "response", "--init_from_teacher", "1"],
                base=base, save_as=f"sweep_s{S:02d}_crown8_t17init_s{seed}")
            add("teachers/crown8-response-bornagain", seed,
                ["--teacher", os.path.join(a.runs_dir, ba_name(seed)), "--montage", "crown8",
                 "--condition", "response"], base=base)
            for c in (["none", "response"] if "teachers" in groups else ["none"]):
                add(f"teachers/crown8-{c}-init63", seed,
                    ["--teacher", T_ALL, "--montage", "crown8", "--condition", c, "--init_from_teacher", "1"],
                    base=base)
        if "progressive" in groups:
            base = crown_base(seed)
            for init in [0, 1]:
                prev = T_ALL
                for m in ["nested32", "nested16", "crown8"]:
                    save = f"sweep_s{S:02d}_takd_init{init}_{m}_s{seed}" if m != "crown8" else None
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
        if "agnostic" in groups:
            # one model for every montage; train_agnostic.py writes a row per montage and the
            # crown8 row is the done-marker. Baselines are the dedicated per-montage students.
            for c, init in [("none", 0), ("response", 0), ("response", 1)]:
                sfx = f"{c}{'-init' if init else ''}"
                add(f"agnostic/crown8-{sfx}", seed,
                    ["--teacher", T_ALL, "--condition", c, "--init_from_teacher", str(init)],
                    script="train_agnostic.py")
        if "pooled" in groups:
            # one training run on every subject's data, one CSV row per test subject
            subs = ",".join(map(str, pool))
            common = ["--subject", str(pool[0]), "--train_subjects", subs, "--eval_subjects", subs, "--teacher", T_POOL]
            add("pooled/crown8-none", seed, common + ["--montage", "crown8", "--condition", "none"],
                base="channels/crown8-none", subject=pool[0])
            add("pooled/crown8-response-init", seed,
                common + ["--montage", "crown8", "--condition", "response", "--init_from_teacher", "1"],
                base="channels/crown8-none", subject=pool[0])
            add("pooled/occipital8-none", seed, common + ["--montage", "occipital8", "--condition", "none"],
                base="channels/occipital8-none", subject=pool[0])
        if "loso" in groups:
            others = ",".join(str(s) for s in pool if s != S)
            lo = ["--train_subjects", others, "--teacher", T_LOSO]
            add("loso/crown8-none", seed, lo + ["--montage", "crown8", "--condition", "none"])
            add("loso/crown8-response-init", seed,
                lo + ["--montage", "crown8", "--condition", "response", "--init_from_teacher", "1"],
                base="loso/crown8-none")
            add("loso/occipital8-none", seed, lo + ["--montage", "occipital8", "--condition", "none"])
            cal = ["--target_frac", "0.1"]
            # floor: only 10% of the subject's own data (smaller batch so it gets >1 update per epoch)
            add("loso/crown8-none-cal10only", seed,
                ["--train_subjects", "none", "--teacher", T_LOSO, "--montage", "crown8", "--condition", "none"] + cal,
                post=["--batch_size", "256"])
            add("loso/crown8-none-cal10", seed, lo + cal + ["--montage", "crown8", "--condition", "none"],
                base="loso/crown8-none-cal10only")
            add("loso/crown8-response-init-cal10", seed,
                lo + cal + ["--montage", "crown8", "--condition", "response", "--init_from_teacher", "1"],
                base="loso/crown8-none-cal10")
    return runs


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def cmd_run(a):
    runs = plan(a)
    done = {(str(r.get("subject")), r.get("tag")) for r in read_rows(a.results_csv)}

    def is_done(r):
        tagged = (str(r["subject"]), f"{r['config']}|seed{r['seed']}") in done
        saved = not r["save_as"] or os.path.exists(os.path.join(a.runs_dir, r["save_as"], "checkpoint_last.pth"))
        return tagged and saved

    todo = [r for r in runs if not is_done(r)]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to run", flush=True)
    for i, r in enumerate(todo, 1):
        tag = f"{r['config']}|seed{r['seed']}"
        args = r["args"] if "--subject" in r["args"] else ["--subject", str(r["subject"])] + r["args"]
        cmd = [sys.executable, os.path.join(ROOT, r["script"]), *args, "--seed", str(r["seed"]),
               "--tag", tag, "--results_csv", a.results_csv, "--runs_dir", a.runs_dir,
               "--epochs", str(a.epochs), *a.extra, *r["post"]]
        if r["save_as"]:
            cmd += ["--save_as", r["save_as"]]
        print(f"[{i}/{len(todo)}] s{r['subject']:02d} {tag}", flush=True)
        if a.dry_run:
            print("   ", " ".join(cmd[1:]))
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-2000:], out.stderr[-3000:])
            raise SystemExit(f"FAILED: s{r['subject']:02d} {tag}")
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
    if a.subjects:
        rows = [r for r in rows if int(r["subject"]) in a.subjects]
    if not rows:
        raise SystemExit(f"no sweep rows in {a.results_csv}")
    a.groups, a.seeds = ["all"], sorted({int(r["seed"]) for r in rows})
    base_of = {r["config"]: r["base"] for r in plan(a)}
    by = defaultdict(dict)  # config -> (subject, seed) -> row
    for r in rows:
        by[r["tag"].split("|seed")[0]][(int(r["subject"]), int(r["seed"]))] = r  # last row wins
    out = []
    for cfg in sorted(by, key=lambda c: (c.split("/")[0], c)):
        cells = by[cfg]
        line = {"config": cfg, "n": len(cells), "subj": len({s for s, _ in cells}),
                "channels": next(iter(cells.values()))["n_channels"]}
        for m in METRICS:
            mu, sd = mean_std([float(r[m]) for r in cells.values() if r.get(m) not in (None, "", "nan")])
            line[m] = f"{mu:.1f} ± {sd:.1f}" if not math.isnan(sd) else f"{mu:.1f}"
        base = base_of.get(cfg)
        if cfg.startswith("agnostic/"):  # vs the dedicated no-KD student on the same montage
            base = f"channels/{cfg.split('/')[1].rsplit('-', 2 if cfg.endswith('-init') else 1)[0]}-none"
        if base and base in by:
            d = [float(cells[k]["top1"]) - float(by[base][k]["top1"]) for k in cells if k in by[base]]
            mu, sd = mean_std(d)
            line["d_top1_vs_base"] = f"{mu:+.1f} ± {sd:.1f}" if len(d) > 1 else (f"{mu:+.1f}" if d else "")
            line["wins"] = f"{sum(x > 0 for x in d)}/{len(d)}"
        out.append(line)
    cols = ["config", "channels", "subj", "n"] + METRICS + ["d_top1_vs_base", "wins"]
    widths = {c: max(len(c), *(len(str(l.get(c, ""))) for l in out)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for l in out:
        print("  ".join(str(l.get(c, "")).ljust(widths[c]) for c in cols))
    print("\nn: (subject, seed) runs; subj: subjects. d_top1_vs_base: paired top-1 gain over the baseline "
          "at the same subject and seed (points); wins: pairs where the gain is > 0. One test image = 0.5 points.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "summary"])
    p.add_argument("--groups", nargs="+", default=["channels"], choices=GROUPS + ["all"])
    p.add_argument("--subject", type=int, default=8, help="target subject for per-subject groups")
    p.add_argument("--subjects", type=int, nargs="+", default=None, help="summary: only these subjects")
    p.add_argument("--pool_subjects", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--teacher_all", default="runs/nb_s{subject:02d}_all")
    p.add_argument("--teacher_post17", default="runs/nb_s{subject:02d}_post17")
    p.add_argument("--teacher_pool", default="runs/nb_pool_post17")
    p.add_argument("--teacher_loso", default="runs/nb_loso{subject:02d}_post17")
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--results_csv", default="results/sweep.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="anything after --extra is passed to every training call")
    a = p.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    main()
