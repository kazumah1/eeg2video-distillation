#!/usr/bin/env bash
# Train a NeuroBridge teacher with the official code (pinned commit), then
# expose it as runs/<NAME> so train_student.py can load it.
#
# Usage: bash scripts/train_teacher_neurobridge.sh <NAME> <SUBJECTS> <CHANNELS> [extra train.py flags]
#   SUBJECTS = one subject (8) or a comma list for a multi-subject teacher (1,2,3,4,5,6,7,9,10)
#   CHANNELS = "all" (63) or "posterior17" (the published configuration)
# Example:
#   bash scripts/train_teacher_neurobridge.sh nb_s08_post17 8 posterior17
#   bash scripts/train_teacher_neurobridge.sh nb_s08_all 8 all
# Env: TEST_SUB (subject NeuroBridge reports on; default: the first of SUBJECTS; it does not
# affect training), FAST_SMOOTH=1 (vectorized smoothing, see scripts/nb_train_fast.py),
# EPOCHS (default 50); PUBLISHED=0 drops the eeg-smoothing / frozen-prior /
# test-image-aug flags of the published intra-subject script (faster load, lower accuracy).
set -euo pipefail
NAME=$1; SUBS=$2; CHANS=$3; shift 3
TRAIN_IDS=${SUBS//,/ }
TEST_SUB=${TEST_SUB:-${TRAIN_IDS%% *}}
FAST_SMOOTH=${FAST_SMOOTH:-0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NB=$ROOT/external/NeuroBridge
NB_COMMIT=4f59fbb9597dbc1a0a9276a984f4f4c8b43ae7d6
DATA=${DATA:-$ROOT/data/things_eeg}
EPOCHS=${EPOCHS:-50}
PUBLISHED=${PUBLISHED:-1}

if [ ! -d "$NB/.git" ]; then
  git clone -q https://github.com/feroooooo/NeuroBridge.git "$NB"
fi
git -C "$NB" fetch -q origin "$NB_COMMIT" 2>/dev/null || true
git -C "$NB" checkout -q "$NB_COMMIT"

if [ "$CHANS" = "posterior17" ]; then
  SEL=(P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2)
elif [ "$CHANS" = "all" ]; then
  SEL=()
else
  echo "CHANNELS must be all or posterior17"; exit 1
fi

AUG_DIR=$DATA/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic
AUG_FLAGS=()
if [ -d "$AUG_DIR" ]; then
  AUG_FLAGS=(--image_aug --aug_image_feature_dirs "$AUG_DIR")
  if [ "$PUBLISHED" = "1" ]; then AUG_FLAGS+=(--image_test_aug); fi
else
  echo "note: $AUG_DIR not found; training without image augmentation (lower than published)"
fi

EEG_FLAGS=()
if [ "$PUBLISHED" = "1" ]; then
  # One-time random temporal smoothing of the train EEG at load (pure-Python loop: a few
  # minutes for 17 channels, longer for 63). Test EEG is not smoothed.
  EEG_FLAGS=(--eeg_aug --eeg_aug_type smooth --frozen_eeg_prior)
fi

OUT=$ROOT/runs/_neurobridge
mkdir -p "$OUT"
# NeuroBridge exits (status 0) if a run with the same name exists, so make every run name
# unique; runs/<NAME> is repointed to the newest run. After a successful run, older runs of
# the same NAME in runs/_neurobridge are deleted (KEEP_OLD=1 keeps them).
RUN_NAME="$NAME-v$(date +%s)"
rm -f "$OUT/last_run.txt"
cd "$NB"
# Mirrors scripts/things_eeg/intra-subjects.sh (PUBLISHED=1). ${a[@]+"${a[@]}"} keeps
# empty arrays safe under set -u on macOS bash 3.2.
if [ "$FAST_SMOOTH" = "1" ]; then TRAIN=("$ROOT/scripts/nb_train_fast.py"); else TRAIN=(train.py); fi
TQDM_DISABLE=1 python "${TRAIN[@]}" \
  --batch_size 1024 --learning_rate 1e-4 --num_epochs "$EPOCHS" \
  --output_dir "$OUT" --output_name "$RUN_NAME" \
  --eeg_encoder_type EEGProject \
  --train_subject_ids $TRAIN_IDS --test_subject_ids "$TEST_SUB" \
  --softplus --img_l2norm --projector linear --feature_dim 512 \
  --image_feature_dir "$DATA/image_feature/RN50" --text_feature_dir "" \
  --eeg_data_dir "$DATA/preprocessed_eeg" \
  --selected_channels ${SEL[@]+"${SEL[@]}"} \
  --data_average --save_weights --seed 2025 --device cuda:0 \
  ${AUG_FLAGS[@]+"${AUG_FLAGS[@]}"} ${EEG_FLAGS[@]+"${EEG_FLAGS[@]}"} "$@"

RUN_DIR=$(cat "$OUT/last_run.txt" 2>/dev/null || true)
case "$RUN_DIR" in
  *"-$RUN_NAME") ;;
  *) echo "ERROR: training did not produce run $RUN_NAME (last_run.txt: '$RUN_DIR')"; exit 1 ;;
esac
mkdir -p "$ROOT/runs"
ln -sfn "$RUN_DIR" "$ROOT/runs/$NAME"
if [ "${KEEP_OLD:-0}" != "1" ] && [ -n "$NAME" ] && [ "$(basename "$OUT")" = "_neurobridge" ]; then
  find "$OUT" -mindepth 1 -maxdepth 1 -type d \( -name "*-$NAME" -o -name "*-$NAME-v*" \) \
    ! -path "$RUN_DIR" -print -exec rm -rf {} + | sed 's/^/removed old run: /'
fi
echo "teacher ready: runs/$NAME -> $RUN_DIR"
cat "$RUN_DIR/result.csv"
