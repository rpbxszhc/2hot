#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/results

python src/pct.py predict \
  --checkpoint outputs/checkpoints/best_model_improved.pkl \
  --batch_size 24 \
  --n_points 1024 \
  --width 160 \
  --emb_dims 1024 \
  --dropout 0.35 \
  --votes 10 \
  --result_path outputs/results/result_improved.json \
  --zip_path outputs/results/submit_improved.zip \
  "$@"
