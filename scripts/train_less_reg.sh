#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/checkpoints outputs/logs outputs/results

python src/pct.py train \
  --epochs 220 \
  --batch_size 24 \
  --n_points 1024 \
  --width 160 \
  --emb_dims 1024 \
  --dropout 0.25 \
  --label_smoothing 0.0 \
  --val_votes 3 \
  --votes 10 \
  --checkpoint outputs/checkpoints/best_model_less_reg.pkl \
  --result_path outputs/results/result_less_reg.json \
  --zip_path outputs/results/submit_less_reg.zip \
  --log_interval 100 \
  "$@" \
  2>&1 | tee outputs/logs/train_less_reg.log
