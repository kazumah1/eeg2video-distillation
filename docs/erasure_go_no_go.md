# Controlled erasure: go/no-go summary (Nov 20)

**Recommendation: GO**, with a modest-effect framing.

## Question

When a frozen NeuroBridge EEG→CLIP decoder retrieves the right image, how much does it rely on
semantic information (object category) versus low-level image statistics (edges, contrast,
spatial frequency)? And does that balance depend on how noisy the EEG is?

## Setup

- THINGS-EEG2, 10 subjects. One 63-channel NeuroBridge teacher per subject (trained locally on
  CPU; reproduces the GPU run). 200-image test gallery, so chance top-1 = 0.005.
- **Intervention:** project out the train cross-covariance span of a property from the EEG
  embedding, before normalization. Probes are refit after each edit to confirm the property was
  removed.
- **Targets:**
  - category: 28 coarse classes, rank 27;
  - low-level: edges, contrast and spatial frequency erased together as one target, rank 3.
- **Noise levels:** test EEG averaged over 1, 4 or 80 repetitions (baseline top-1 0.05 / 0.15 / 0.55).
- **Controls, weakest to strictest:**
  1. random subspace of the same rank and per-image L2;
  2. 10 concept-shuffled copies of the label;
  3. those shuffled copies rescaled to the real edit's per-image L2 (size-matched).
- The analysis settings and the 0.02 smallest effect of interest were frozen on the sub-08 pilot.

## Result

The table shows extra top-1 damage beyond the size-matched shuffled controls, with the mean
and 95% bootstrap CI over the 10 subjects.

| Repetitions | Low-level | Category | Low-level − category (paired) |
|---|---|---|---|
| 1 | +0.006 [−0.000, 0.012] | +0.006 [0.004, 0.009] | 0.000 [−0.008, 0.007] |
| 4 | **+0.023** [0.017, 0.030], 10/10 subjects | +0.017 [0.011, 0.023], 9/10 | +0.006 [−0.002, 0.014] |
| 80 | **+0.044** [0.034, 0.055], 10/10 | **+0.064** [0.049, 0.081], 10/10 | −0.020 [−0.034, −0.009] |

![crossover](../results/figures/crossover.png)

1. **Naive controls overstate semantic reliance about 5×.**
   - At 80 repetitions, category damage falls from +0.336 (random controls) to +0.132 (shuffled)
     to +0.064 (size-matched).
   - Any data-fitted erasure removes concept-identity structure. Random subspaces do not capture
     that, so they inflate the effect.
2. **The decoder uses a separable low-level family.**
   - Erasing the three low-level descriptors removes them (the refit probes drop to chance) and
     hurts retrieval beyond every control.
   - It leaves category information nearly intact at 4 and 80 repetitions: 1–14% extra loss, below the 25% collateral
     tolerance.
   - The reverse does not hold: erasing category also strips about half the low-level
     information.
3. **The balance depends on noise.**
   - With noisy EEG (4 repetitions), low-level damage is as large as category damage: 15% vs 11%
     of baseline.
   - With clean EEG (80 repetitions), category leads: 12% vs 8%.

![control ladder](../results/figures/control_ladder.png)

## Caveats

- **The effects are small.** At 4 repetitions the low-level mean clears 0.02, but the lower end
  of its CI (0.017) does not.
- **At 1 repetition everything is below 0.02.** That noise level supports no claim either way.
- **Low-level means the family, not single descriptors.** At 80 repetitions, edges alone and
  spatial frequency alone fall to about +0.01 under size-matched controls. The descriptors are
  too entangled to separate.
- **Size matching is a conservative control.** The shuffled edit is pushed past its own
  projection, which can overstate generic damage. Results under both shuffled controls are shown.
- **Untested so far:**
  - only one decoder architecture (NeuroBridge, CLIP RN50 targets);
  - only one coarse category scheme;
  - no electrode-montage contrast yet.

## Talk headline

> Strict controls cut apparent semantic reliance in EEG image decoding about 5×. What remains is
> semantic plus low-level image statistics, and when the EEG is noisy the low-level share is as
> large as the semantic one.

## Next steps

1. Reconcile this branch with `codex/controlled-erasure` before merging.
2. Optional: the crown-8 vs occipital-8 montage contrast (tooling exists: `mi_verdict.py --vs`).
3. Turn the two figures into talk slides.

## Reproduce

```bash
ANNOT=$PWD/data/things_eeg/image_set/annotations.csv OUT=$PWD/results/erasure_sm \
  bash scripts/erasure_all_subjects.sh
python mi_verdict.py results/erasure_sm/sub-*.json --sesoi 0.02 \
  --probe-chance coarse_category=0.0357 --size-matched > results/verdicts_sm_10sub.json
python scripts/plot_erasure_summary.py results/verdicts_sm_10sub.json
```

The code is at commit `080bfb4` on branch `claude/superpowers-tdd-review-70a7a1`. The full run
takes about 2 hours on an M4 Pro CPU.
