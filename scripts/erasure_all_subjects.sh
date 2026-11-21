#!/usr/bin/env bash
# Official controlled-erasure run: for each subject, download its EEG, train the 63-channel
# NeuroBridge teacher, run mi_erasure.py with the frozen settings below, and write
# $OUT/sub-XX.json. Resumable: subjects whose JSON exists are skipped. Built for a Colab GPU
# runtime (see README); set OUT to a Drive folder so results survive the runtime.
#
# Usage: ANNOT=/content/drive/MyDrive/eegdistillation/annotations.csv \
#        OUT=/content/drive/MyDrive/eegdistillation/erasure \
#        bash scripts/erasure_all_subjects.sh [SUBJECT ...]       # default: 1..10
# Env: DATA (default $ROOT/data/things_eeg; never point it at a copy you need to keep, since
#      each subject's EEG, ~4.8 GB, is deleted after its run unless KEEP_EEG=1);
#      RUNS (default $ROOT/runs) holds teacher and student checkpoints;
#      DRY_RUN=1 prints the commands only; PY (default python);
#      MONTAGES="crown8 occipital8 crown8@response" trains a student per entry (no KD unless
#      @CONDITION) and writes $OUT/<montage>[_<condition>]/sub-XX.json (default "all": the 63-channel teacher, $OUT/sub-XX.json).
# Then: python mi_verdict.py $OUT/sub-*.json --sesoi 0.02 --probe-chance coarse_category=0.0357
#       [--size-matched]  (shuffled copies are also scored at the real edit's per-image size)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${ANNOT:?set ANNOT to the aligned annotations CSV}"
OUT=${OUT:-$ROOT/results/erasure}
PY=${PY:-python}
RUNS=${RUNS:-$ROOT/runs}
export DATA=${DATA:-$ROOT/data/things_eeg}
FEAT=$DATA/image_feature/RN50
SUBJECTS=("$@"); [ ${#SUBJECTS[@]} -gt 0 ] || SUBJECTS=(1 2 3 4 5 6 7 8 9 10)

# Frozen analysis settings (decided on the sub-08 pilot, before any other subject was seen;
# the joint low-level target was added after the first 10-subject run showed the three
# descriptors are not separable).
ERASURE_FLAGS=(
  --eeg-dir "$DATA/preprocessed_eeg" --feat-dir "$FEAT"
  --aug-dir "$FEAT/GaussianBlur-GaussianNoise-LowResolution-Mosaic" --eval-img augmean
  --annotations "$ANNOT"
  --descriptor edge_density --descriptor contrast --descriptor spatial_frequency
  --category coarse_category --residualize edge_density --rank-matched-category
  --joint lowlevel=edge_density,contrast,spatial_frequency
  --shuffled-controls 10 --controls 20 --strength 1.0 --ridge 1e-4 --seed 0
  --repeats 1 4 all
)

run() { echo "+ $*"; [ "${DRY_RUN:-0}" = "1" ] || "$@"; }

MONTAGES=(${MONTAGES:-all})
# A montage entry is MONTAGE or MONTAGE@CONDITION (a train_student.py preset, default none).
name_of() { case "$1" in *@none) echo "${1%@*}";; *@*) echo "${1%@*}_${1#*@}";; *) echo "$1";; esac; }
dest_of() { if [ "$2" = all ]; then echo "$OUT/sub-$1.json"; else echo "$OUT/$(name_of "$2")/sub-$1.json"; fi; }

mkdir -p "$OUT"
for s in "${SUBJECTS[@]}"; do
  sub=$(printf "%02d" "$s")
  todo=()
  for m in "${MONTAGES[@]}"; do
    dest=$(dest_of "$sub" "$m")
    if [ -s "$dest" ]; then echo "skip sub-$sub $m: $dest exists"; else todo+=("$m"); fi
  done
  [ ${#todo[@]} -gt 0 ] || continue
  # Both splits must be present: an interrupted download can leave test.npy alone.
  if [ ! -e "$DATA/preprocessed_eeg/sub-$sub/test.npy" ] || [ ! -e "$DATA/preprocessed_eeg/sub-$sub/train.npy" ]; then
    run "$PY" "$ROOT/scripts/download_things_eeg.py" --subjects "$s" --out "$(dirname "$DATA")"
  fi
  teacher=$RUNS/nb_s${sub}_all
  if [ ! -e "$teacher/checkpoint_last.pth" ]; then
    run bash "$ROOT/scripts/train_teacher_neurobridge.sh" "nb_s${sub}_all" "$s" all
  fi
  for m in "${todo[@]}"; do
    dest=$(dest_of "$sub" "$m")
    model=$teacher
    if [ "$m" != all ]; then
      # Student on this montage (no distillation unless @CONDITION); it reuses the teacher's
      # frozen image projector, so every model is scored in the same embedding space and gallery.
      montage=${m%@*}; condition=none
      case "$m" in *@*) condition=${m#*@};; esac
      model=$RUNS/nb_s${sub}_$(name_of "$m")
      if [ ! -e "$model/checkpoint_last.pth" ]; then
        run "$PY" "$ROOT/train_student.py" --teacher "$teacher" --eeg_dir "$DATA/preprocessed_eeg" \
          --feat_dir "$FEAT" --aug_dir "$FEAT/GaussianBlur-GaussianNoise-LowResolution-Mosaic" --montage "$montage" --condition "$condition" --subject "$s" --save_as "nb_s${sub}_$(name_of "$m")" \
          --runs_dir "$RUNS" --results_csv "$OUT/students.csv" --device "${DEVICE:-cpu}"
      fi
    fi
    mkdir -p "$(dirname "$dest")"
    # Write to a temp file first so an interrupted run never leaves a JSON that resume would skip.
    if [ "${DRY_RUN:-0}" = "1" ]; then
      echo "+ $PY mi_erasure.py --run $s $model ${ERASURE_FLAGS[*]} > $dest"
    else
      "$PY" "$ROOT/mi_erasure.py" --run "$s" "$model" "${ERASURE_FLAGS[@]}" \
        > "$dest.part" && mv "$dest.part" "$dest"
    fi
  done
  if [ "${KEEP_EEG:-0}" != "1" ]; then run rm -rf "$DATA/preprocessed_eeg/sub-$sub"; fi
done
echo "done: $(ls "$OUT"/sub-*.json "$OUT"/*/sub-*.json 2>/dev/null | wc -l | tr -d ' ') reports in $OUT"
