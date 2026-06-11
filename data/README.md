# Data

Place the competition-provided ModelNet40 point-cloud files in this directory.

Expected structure:

```text
data/
  train_points.npy   # (9843, 2048, 3), float32
  train_labels.npy   # (9843,), int32, labels 0-39
  test_points.npy    # (2468, 2048, 3), float32
  categories.txt     # 40 category names
```

If you have `data.zip`, unzip it from the project root:

```bash
unzip -o data/data.zip -d .
```

Large data files are intentionally ignored by git.
