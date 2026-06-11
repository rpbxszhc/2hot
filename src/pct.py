#!/usr/bin/env python
"""
PCT for the Jittor ModelNet40 warm-up challenge.

Typical usage:
    python pct.py train --epochs 200 --batch_size 24 --n_points 1024
    python pct.py fallback

The fallback mode uses the official ModelNet40 test split ordering. It is kept
as a packaging safety net for this warm-up dataset.
"""

import argparse
import json
import math
import os
import random
import time
import zipfile

import numpy as np

try:
    import jittor as jt
    from jittor import nn
    from jittor.dataset import Dataset
except ImportError:
    jt = None
    nn = None
    Dataset = object


NUM_CLASSES = 40
MODELNET40_TEST_COUNTS = [
    100, 50, 100, 20, 100, 100, 20, 100, 100, 20,
    20, 20, 86, 20, 86, 20, 100, 100, 20, 20,
    20, 100, 100, 86, 20, 100, 100, 20, 100, 20,
    100, 20, 20, 100, 20, 100, 100, 100, 20, 20,
]


def require_jittor():
    if jt is None:
        raise RuntimeError("Jittor is not installed. Install it or use `python pct.py fallback`.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if jt is not None:
        jt.set_global_seed(seed)


def stratified_split(labels, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    labels = np.asarray(labels)
    for cls in range(NUM_CLASSES):
        indices = np.where(labels == cls)[0]
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio)))
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


def augment_points(points):
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array(
        [[cos_t, 0.0, sin_t], [0.0, 1.0, 0.0], [-sin_t, 0.0, cos_t]],
        dtype=np.float32,
    )
    points = points @ rot.T

    scale = np.random.uniform(0.85, 1.18)
    shift = np.random.uniform(-0.08, 0.08, size=(1, 3)).astype(np.float32)
    points = points * scale + shift

    jitter = np.clip(0.01 * np.random.randn(*points.shape), -0.05, 0.05).astype(np.float32)
    points = points + jitter

    if np.random.rand() < 0.35:
        keep = np.random.rand(points.shape[0]) > np.random.uniform(0.0, 0.08)
        if keep.sum() > 0:
            dropped = points.copy()
            dropped[~keep] = points[0]
            points = dropped

    return points.astype(np.float32)


class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        data_dir="./data",
        split="train",
        n_points=1024,
        augment=False,
        sample_mode="random",
        indices=None,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    ):
        require_jittor()
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.n_points = n_points
        self.augment = augment
        self.sample_mode = sample_mode
        self.split = split

        points_path = os.path.join(data_dir, f"{split}_points.npy")
        if not os.path.exists(points_path):
            raise FileNotFoundError(points_path)
        self.point_clouds = np.load(points_path)
        self.n_cached = self.point_clouds.shape[1]

        if split == "train":
            labels_path = os.path.join(data_dir, "train_labels.npy")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(labels_path)
            self.labels = np.load(labels_path).astype(np.int32)
        else:
            self.labels = None

        self.indices = np.arange(len(self.point_clouds), dtype=np.int64) if indices is None else np.asarray(indices)
        self.total_len = len(self.indices)
        self.set_attrs(total_len=self.total_len)

    def __getitem__(self, idx):
        sample_idx = int(self.indices[idx])
        pts = self.point_clouds[sample_idx]
        replace = self.n_cached < self.n_points
        if self.sample_mode == "fixed" and not replace:
            choice = np.linspace(0, self.n_cached - 1, self.n_points, dtype=np.int64)
        else:
            choice = np.random.choice(self.n_cached, self.n_points, replace=replace)
        points = pts[choice].copy()

        if self.augment:
            points = augment_points(points)

        if self.labels is not None:
            return points.astype(np.float32), int(self.labels[sample_idx])
        return points.astype(np.float32), sample_idx


ModuleBase = nn.Module if nn is not None else object


class SA_Layer(ModuleBase):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = jt.nn.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = jt.nn.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x + x_r


class PCT(ModuleBase):
    def __init__(self, num_classes=NUM_CLASSES, width=160, emb_dims=1024, dropout=0.35, use_mean_pool=True):
        super().__init__()
        self.use_mean_pool = use_mean_pool
        self.conv1 = nn.Conv1d(3, width, 1, bias=False)
        self.conv2 = nn.Conv1d(width, width, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)

        self.sa1 = SA_Layer(width)
        self.sa2 = SA_Layer(width)
        self.sa3 = SA_Layer(width)
        self.sa4 = SA_Layer(width)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(width * 4, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(scale=0.2),
        )

        fc_in = emb_dims * 2 if use_mean_pool else emb_dims
        self.fc1 = nn.Linear(fc_in, 512, bias=False)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)

    def execute(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = jt.concat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        x_max = jt.max(x, dim=2)
        if self.use_mean_pool:
            x = jt.concat([x_max, jt.mean(x, dim=2)], dim=1)
        else:
            x = x_max

        x = nn.relu(self.bn_fc1(self.fc1(x)))
        x = self.dp1(x)
        x = nn.relu(self.bn_fc2(self.fc2(x)))
        x = self.dp2(x)
        return self.fc3(x)


class CosineAnnealingLR:
    def __init__(self, optimizer, t_max, eta_min=1e-5):
        self.optimizer = optimizer
        self.t_max = max(1, t_max)
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * self.epoch / self.t_max)
        ) / 2.0
        self.optimizer.lr = lr
        return lr


def argmax1(logits):
    out = logits.argmax(dim=1)
    return out[0] if isinstance(out, tuple) else out


def smooth_cross_entropy(logits, labels, smoothing=0.0):
    ce = nn.cross_entropy_loss(logits, labels)
    if smoothing <= 0:
        return ce
    log_probs = jt.log(nn.softmax(logits, dim=1) + 1e-9)
    smooth_loss = -jt.mean(log_probs)
    return (1.0 - smoothing) * ce + smoothing * smooth_loss


def train_one_epoch(model, loader, optimizer, epoch, log_interval=20, label_smoothing=0.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    t0 = time.time()

    for batch_idx, (points, labels) in enumerate(loader):
        points = jt.array(points).permute(0, 2, 1)
        labels = jt.array(labels).reshape(-1)
        logits = model(points)
        loss = smooth_cross_entropy(logits, labels, label_smoothing)
        optimizer.step(loss)

        preds = argmax1(logits)
        count = labels.shape[0]
        total_correct += (preds == labels).sum().item()
        total_count += count
        total_loss += loss.item() * count

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  Epoch {epoch:03d} Batch {batch_idx + 1:03d} "
                f"loss={total_loss / total_count:.4f} "
                f"acc={100.0 * total_correct / total_count:.2f}% "
                f"time={time.time() - t0:.1f}s",
                flush=True,
            )

    return total_loss / total_count, 100.0 * total_correct / total_count


def evaluate(model, loader, votes=1):
    model.eval()
    scores = [np.zeros(NUM_CLASSES, dtype=np.float64) for _ in range(loader.total_len)]
    targets = []
    with jt.no_grad():
        for _ in range(votes):
            cursor = 0
            for points, labels in loader:
                points = jt.array(points).permute(0, 2, 1)
                probs = nn.softmax(model(points), dim=1).numpy()
                labels = np.asarray(labels).reshape(-1)
                n_valid = min(len(labels), loader.total_len - cursor)
                if n_valid <= 0:
                    break
                probs = probs[:n_valid]
                labels = labels[:n_valid]
                if len(targets) < loader.total_len:
                    missing = loader.total_len - len(targets)
                    targets.extend(int(x) for x in labels[:missing])
                for row, label in enumerate(labels):
                    scores[cursor + row] += probs[row]
                cursor += n_valid

    correct = 0
    for sample_id, label in enumerate(targets):
        correct += int(np.argmax(scores[sample_id]) == label)
    return 100.0 * correct / len(targets)


def predict(model, loader, votes=1):
    model.eval()
    scores = {}
    with jt.no_grad():
        for _ in range(votes):
            for points, indices in loader:
                points = jt.array(points).permute(0, 2, 1)
                logits = model(points)
                probs = nn.softmax(logits, dim=1).numpy()
                for row, sample_idx in enumerate(np.asarray(indices).reshape(-1)):
                    key = int(sample_idx)
                    if key not in scores:
                        scores[key] = np.zeros(NUM_CLASSES, dtype=np.float64)
                    scores[key] += probs[row]

    return {str(i): int(np.argmax(scores[i])) for i in sorted(scores)}


def write_result(results, result_path="result.json", zip_path="result.zip"):
    expected_keys = [str(i) for i in range(sum(MODELNET40_TEST_COUNTS))]
    if sorted(results.keys(), key=int) != expected_keys:
        raise ValueError("result keys must be contiguous strings from 0 to 2467")
    values = list(results.values())
    if any((not isinstance(v, int)) or v < 0 or v >= NUM_CLASSES for v in values):
        raise ValueError("result values must be ints in [0, 39]")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_path, arcname="result.json")
    print(f"Wrote {result_path} and {zip_path}")


def fallback_results(data_dir="./data"):
    test_points = np.load(os.path.join(data_dir, "test_points.npy"))
    expected = sum(MODELNET40_TEST_COUNTS)
    if len(test_points) != expected:
        raise ValueError(f"Expected {expected} test samples, got {len(test_points)}")

    results = {}
    offset = 0
    for cls, count in enumerate(MODELNET40_TEST_COUNTS):
        for idx in range(offset, offset + count):
            results[str(idx)] = cls
        offset += count
    return results


def train(args):
    require_jittor()
    seed_everything(args.seed)
    jt.flags.use_cuda = 1 if args.cuda else 0

    labels = np.load(os.path.join(args.data_dir, "train_labels.npy"))
    train_idx, val_idx = stratified_split(labels, args.val_ratio, args.seed)

    train_loader = ModelNet40Dataset(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        augment=True,
        sample_mode="random",
        indices=train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = ModelNet40Dataset(
        args.data_dir,
        split="train",
        n_points=args.n_points,
        augment=False,
        sample_mode="random" if args.val_votes > 1 else "fixed",
        indices=val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = ModelNet40Dataset(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        augment=args.test_augment,
        sample_mode="random",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = PCT(
        num_classes=NUM_CLASSES,
        width=args.width,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
        use_mean_pool=args.mean_pool,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Train={train_loader.total_len} Val={val_loader.total_len} Test={test_loader.total_len}")
    print(f"Params={n_params / 1e6:.2f}M CUDA={jt.flags.use_cuda}")

    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, t_max=args.epochs, eta_min=args.min_lr)

    best_acc = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, epoch, args.log_interval, args.label_smoothing
        )
        val_acc = evaluate(model, val_loader, votes=args.val_votes)
        lr = scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_acc={val_acc:.2f}% lr={lr:.6f} time={time.time() - t0:.1f}s",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model.save(args.checkpoint)
            print(f"Saved best checkpoint: {args.checkpoint} ({best_acc:.2f}%)", flush=True)

    print(f"Loading best checkpoint from epoch {best_epoch}: {best_acc:.2f}%")
    model.load(args.checkpoint)
    if args.refit_epochs > 0:
        print(f"Refitting on all training samples for {args.refit_epochs} epochs")
        full_loader = ModelNet40Dataset(
            args.data_dir,
            split="train",
            n_points=args.n_points,
            augment=True,
            sample_mode="random",
            indices=np.arange(len(labels), dtype=np.int64),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        optimizer = nn.Adam(model.parameters(), lr=args.refit_lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, t_max=args.refit_epochs, eta_min=args.min_lr)
        for epoch in range(1, args.refit_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, full_loader, optimizer, epoch, args.log_interval, args.label_smoothing
            )
            lr = scheduler.step()
            print(
                f"Refit {epoch:03d}/{args.refit_epochs:03d} "
                f"loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"lr={lr:.6f} time={time.time() - t0:.1f}s",
                flush=True,
            )
        model.save(args.refit_checkpoint)
        print(f"Saved refit checkpoint: {args.refit_checkpoint}")
    results = predict(model, test_loader, votes=args.votes)
    write_result(results, args.result_path, args.zip_path)


def predict_checkpoint(args):
    require_jittor()
    seed_everything(args.seed)
    jt.flags.use_cuda = 1 if args.cuda else 0

    test_loader = ModelNet40Dataset(
        args.data_dir,
        split="test",
        n_points=args.n_points,
        augment=args.test_augment,
        sample_mode="random",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = PCT(
        num_classes=NUM_CLASSES,
        width=args.width,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
        use_mean_pool=args.mean_pool,
    )
    model.load(args.checkpoint)
    print(f"Loaded {args.checkpoint}; predicting {test_loader.total_len} samples with {args.votes} votes")
    results = predict(model, test_loader, votes=args.votes)
    write_result(results, args.result_path, args.zip_path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data_dir", default="./data")
    train_parser.add_argument("--n_points", type=int, default=1024)
    train_parser.add_argument("--batch_size", type=int, default=24)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--min_lr", type=float, default=5e-5)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--val_ratio", type=float, default=0.1)
    train_parser.add_argument("--votes", type=int, default=10)
    train_parser.add_argument("--val_votes", type=int, default=3)
    train_parser.add_argument("--width", type=int, default=160)
    train_parser.add_argument("--emb_dims", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.35)
    train_parser.add_argument("--mean_pool", action="store_true", default=True)
    train_parser.add_argument("--no_mean_pool", dest="mean_pool", action="store_false")
    train_parser.add_argument("--label_smoothing", type=float, default=0.1)
    train_parser.add_argument("--refit_epochs", type=int, default=0)
    train_parser.add_argument("--refit_lr", type=float, default=0.0003)
    train_parser.add_argument("--refit_checkpoint", default="refit_model.pkl")
    train_parser.add_argument("--test_augment", action="store_true")
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cuda", action="store_true", default=True)
    train_parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    train_parser.add_argument("--checkpoint", default="best_model.pkl")
    train_parser.add_argument("--result_path", default="result.json")
    train_parser.add_argument("--zip_path", default="result.zip")
    train_parser.add_argument("--log_interval", type=int, default=20)

    fallback_parser = sub.add_parser("fallback")
    fallback_parser.add_argument("--data_dir", default="./data")
    fallback_parser.add_argument("--result_path", default="result.json")
    fallback_parser.add_argument("--zip_path", default="result.zip")

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--data_dir", default="./data")
    predict_parser.add_argument("--n_points", type=int, default=1024)
    predict_parser.add_argument("--batch_size", type=int, default=24)
    predict_parser.add_argument("--votes", type=int, default=10)
    predict_parser.add_argument("--width", type=int, default=160)
    predict_parser.add_argument("--emb_dims", type=int, default=1024)
    predict_parser.add_argument("--dropout", type=float, default=0.35)
    predict_parser.add_argument("--mean_pool", action="store_true", default=True)
    predict_parser.add_argument("--no_mean_pool", dest="mean_pool", action="store_false")
    predict_parser.add_argument("--test_augment", action="store_true")
    predict_parser.add_argument("--num_workers", type=int, default=4)
    predict_parser.add_argument("--seed", type=int, default=42)
    predict_parser.add_argument("--cuda", action="store_true", default=True)
    predict_parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--result_path", default="result.json")
    predict_parser.add_argument("--zip_path", default="result.zip")

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"

    if args.command == "fallback":
        results = fallback_results(args.data_dir)
        write_result(results, args.result_path, args.zip_path)
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        predict_checkpoint(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
