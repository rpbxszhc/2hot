#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/checkpoints outputs/logs outputs/results

python src/pct_full.py train \
  --epochs 120 \
  --batch_size 16 \
  --n_points 1024 \
  --tokens 128 \
  --k 16 \
  --width 160 \
  --emb_dims 1024 \
  --dropout 0.35 \
  --label_smoothing 0.1 \
  --val_votes 3 \
  --votes 10 \
  --checkpoint outputs/checkpoints/best_model_full_pct.pkl \
  --result_path outputs/results/result_full_pct.json \
  --zip_path outputs/results/submit_full_pct.zip \
  --log_interval 100 \
  "$@" \
  2>&1 | tee outputs/logs/train_full_pct.log
