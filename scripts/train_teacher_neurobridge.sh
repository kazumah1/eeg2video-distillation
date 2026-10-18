#!/usr/bin/env bash
# Train a NeuroBridge teacher with the official code (pinned commit), then
# expose it as runs/<NAME> so train_student.py can load it.
#
# Usage: bash scripts/train_teacher_neurobridge.sh <NAME> <SUBJECT> <CHANNELS> [extra train.py flags]
#   CHANNELS = "all" (63) or "posterior17" (the published configuration)
# Example:
#   bash scripts/train_teacher_neurobridge.sh nb_s08_post17 8 posterior17
#   bash scripts/train_teacher_neurobridge.sh nb_s08_all 8 all
set -euo pipefail
NAME=$1; SUB=$2; CHANS=$3; shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NB=$ROOT/external/NeuroBridge
NB_COMMIT=4f59fbb9597dbc1a0a9276a984f4f4c8b43ae7d6
DATA=${DATA:-$ROOT/data/things_eeg}
EPOCHS=${EPOCHS:-50}

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
else
  echo "note: $AUG_DIR not found; training without image augmentation (lower than published)"
fi

OUT=$ROOT/runs/_neurobridge
mkdir -p "$OUT"
cd "$NB"
# Mirrors scripts/things_eeg/intra-subjects.sh except: --eeg_aug/--frozen_eeg_prior are
# opt-in via extra flags (they are slow), and --text_feature_dir is empty.
python train.py \
  --batch_size 1024 --learning_rate 1e-4 --num_epochs "$EPOCHS" \
  --output_dir "$OUT" --output_name "$NAME" \
  --eeg_encoder_type EEGProject \
  --train_subject_ids "$SUB" --test_subject_ids "$SUB" \
  --softplus --img_l2norm --projector linear --feature_dim 512 \
  --image_feature_dir "$DATA/image_feature/RN50" --text_feature_dir "" \
  --eeg_data_dir "$DATA/preprocessed_eeg" \
  --selected_channels "${SEL[@]}" \
  --data_average --save_weights --seed 2025 --device cuda:0 \
  "${AUG_FLAGS[@]}" "$@"

RUN_DIR=$(cat "$OUT/last_run.txt")
mkdir -p "$ROOT/runs"
ln -sfn "$RUN_DIR" "$ROOT/runs/$NAME"
echo "teacher ready: runs/$NAME -> $RUN_DIR"
cat "$RUN_DIR/result.csv"
