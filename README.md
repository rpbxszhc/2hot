# Jittor ModelNet40 Point Cloud Classification

This repository contains code for the Jittor warm-up challenge on ModelNet40
point-cloud classification. The official submission is a zip file containing a
single `result.json` mapping test sample ids to class ids.

## Environment

Recommended environment:

- Python 3.10
- Jittor 1.3.11 or newer
- CUDA-capable GPU recommended

Install with conda:

```bash
conda create -n jittor-hot python=3.10 -y
conda activate jittor-hot
python -m pip install -r requirements.txt
```

Optional Jittor checks:

```bash
python -m jittor.test.test_example
```

## Data

Download the competition-provided ModelNet40 point-cloud data and place it under
`data/`. Expected files:

```text
data/train_points.npy
data/train_labels.npy
data/test_points.npy
data/categories.txt
```

If the downloaded file is `data/data.zip`, unzip it from the project root:

```bash
unzip -o data/data.zip -d .
```

The large data files are ignored by git. See `data/README.md`.

## Training

Primary model used for the best local validation result:

```bash
bash scripts/train_improved.sh
```

This writes checkpoints, logs, and submission artifacts under `outputs/`:

```text
outputs/checkpoints/best_model_improved.pkl
outputs/logs/train_improved.log
outputs/results/submit_improved.zip
```

Alternative runs:

```bash
bash scripts/train_less_reg.sh
bash scripts/train_full_pct.sh
```

All scripts accept extra CLI overrides. Example:

```bash
bash scripts/train_improved.sh --seed 7 --votes 20
```

## Inference

Generate a submission zip from a trained improved checkpoint:

```bash
bash scripts/predict_improved.sh
```

Validate the zip structure before submitting:

```bash
python tools/validate_submission.py outputs/results/submit_improved.zip
```

The zip must contain exactly:

```text
result.json
```

## Results

Metric: classification accuracy on the hidden test labels.

Local validation summary from current experiments:

| Run | Checkpoint | Best val acc |
| --- | --- | ---: |
| improved PCT | `best_model_improved.pkl` | 82.13% |
| less regularized PCT | `best_model_less_reg.pkl` | 81.62% |
| full PCT variant | `best_model_full_pct.pkl` | 81.12% |
| baseline PCT | `best_model_1024.pkl` | 80.10% |

Validation uses a stratified 90/10 train/val split and may differ from the
online hidden-test score because the official test labels are not available.

## Repository Layout

```text
configs/       # reference hyperparameter configs
data/          # data placement instructions; large files are ignored
scripts/       # train and inference entrypoints
src/           # dataset, models, training, inference, packaging
tools/         # validation utilities
outputs/       # local logs, checkpoints, submissions; ignored by git
```

## Reproducibility Notes

- All training entrypoints expose `--seed` and set Python, NumPy, and Jittor
  seeds.
- Key hyperparameters are controlled by CLI arguments and mirrored in
  `configs/`.
- Logs, checkpoints, and result zips are written to `outputs/` by default.
