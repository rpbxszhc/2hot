#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/checkpoints outputs/logs outputs/results

python src/pct.py train \
  --epochs 220 \
  --batch_size 24 \
  --n_points 1024 \
  --width 160 \
  --emb_dims 1024 \
  --dropout 0.35 \
  --label_smoothing 0.1 \
  --val_votes 3 \
  --votes 10 \
  --checkpoint outputs/checkpoints/best_model_improved.pkl \
  --result_path outputs/results/result_improved.json \
  --zip_path outputs/results/submit_improved.zip \
  --log_interval 100 \
  "$@" \
  2>&1 | tee outputs/logs/train_improved.log
