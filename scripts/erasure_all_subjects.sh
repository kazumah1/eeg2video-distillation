#!/usr/bin/env bash
# Official controlled-erasure run: for each subject, download its EEG, train the 63-channel
# NeuroBridge teacher, run mi_erasure.py with the frozen settings below, and write
# $OUT/sub-XX.json. Resumable: subjects whose JSON exists are skipped. Built for a Colab GPU
# runtime (see README); set OUT to a Drive folder so results survive the runtime.
#
# Usage: ANNOT=/content/drive/MyDrive/eegdistillation/annotations.csv \
#        OUT=/content/drive/MyDrive/eegdistillation/erasure \
#        bash scripts/erasure_all_subjects.sh [SUBJECT ...]       # default: 1..10
# Env: DRY_RUN=1 prints the commands only; KEEP_EEG=1 keeps each subject's EEG (~4.8 GB)
#      after its run; PY (default python).
# Then: python mi_verdict.py $OUT/sub-*.json --sesoi 0.02 --probe-chance coarse_category=0.0357
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${ANNOT:?set ANNOT to the aligned annotations CSV}"
OUT=${OUT:-$ROOT/results/erasure}
PY=${PY:-python}
DATA=$ROOT/data/things_eeg
FEAT=$DATA/image_feature/RN50
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -gt 0 ] || SUBJECTS=(1 2 3 4 5 6 7 8 9 10)

# Frozen analysis settings (decided on the sub-08 pilot, before any other subject was seen).
ERASURE_FLAGS=(
  --eeg-dir "$DATA/preprocessed_eeg" --feat-dir "$FEAT"
  --aug-dir "$FEAT/GaussianBlur-GaussianNoise-LowResolution-Mosaic" --eval-img augmean
  --annotations "$ANNOT"
  --descriptor edge_density --descriptor contrast --descriptor spatial_frequency
  --category coarse_category --residualize edge_density --rank-matched-category
  --shuffled-controls 10 --controls 20 --strength 1.0 --ridge 1e-4 --seed 0
  --repeats 1 4 all
)

run() { echo "+ $*"; [ "${DRY_RUN:-0}" = "1" ] || "$@"; }

mkdir -p "$OUT"
for s in "${SUBJECTS[@]}"; do
  sub=$(printf "%02d" "$s")
  dest=$OUT/sub-$sub.json
  if [ -s "$dest" ]; then echo "skip sub-$sub: $dest exists"; continue; fi
  run "$PY" "$ROOT/scripts/download_things_eeg.py" --subjects "$s" --out "$ROOT/data"
  if [ ! -e "$ROOT/runs/nb_s${sub}_all/checkpoint_last.pth" ]; then
    run bash "$ROOT/scripts/train_teacher_neurobridge.sh" "nb_s${sub}_all" "$s" all
  fi
  # Write to a temp file first so an interrupted run never leaves a JSON that resume would skip.
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "+ $PY mi_erasure.py --run $s runs/nb_s${sub}_all ${ERASURE_FLAGS[*]} > $dest"
  else
    "$PY" "$ROOT/mi_erasure.py" --run "$s" "$ROOT/runs/nb_s${sub}_all" "${ERASURE_FLAGS[@]}" \
      > "$dest.part" && mv "$dest.part" "$dest"
  fi
  if [ "${KEEP_EEG:-0}" != "1" ]; then run rm -rf "$DATA/preprocessed_eeg/sub-$sub"; fi
done
echo "done: $(ls "$OUT"/sub-*.json 2>/dev/null | wc -l | tr -d ' ') subject reports in $OUT"
