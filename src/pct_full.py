#!/usr/bin/env python
"""
Fuller PCT-style model for the Jittor ModelNet40 warm-up challenge.

This file intentionally does not replace pct.py. It keeps the same dataset,
training loop style, result.json writer, and zip packaging, but swaps the model
for a stronger local-neighborhood + offset-attention architecture.

Typical usage:
    python pct_full.py train --epochs 220 --batch_size 12 --n_points 1024
"""

import argparse
import os
import time

import numpy as np

from pct import (
    NUM_CLASSES,
    CosineAnnealingLR,
    ModelNet40Dataset,
    evaluate,
    predict,
    require_jittor,
    seed_everything,
    smooth_cross_entropy,
    stratified_split,
    write_result,
)
from pct import jt, nn


def get_graph_feature(x, k=20):
    """Build EdgeConv features.

    Args:
        x: (B, C, N)
    Returns:
        (B, 2C, N, k), concat(neighbor - center, center)
    """
    B, C, N = x.shape
    x_t = x.permute(0, 2, 1)
    inner = -2.0 * jt.nn.bmm(x_t, x)
    xx = jt.sum(x * x, dim=1, keepdims=True)
    pairwise_distance = -xx.permute(0, 2, 1) - inner - xx
    _, idx = jt.topk(pairwise_distance, k=k, dim=-1)

    neighbors = x_t.reindex(
        [B, N, k, C],
        ["i0", "@e0(i0,i1,i2)", "i3"],
        extras=[idx],
    )
    centers = x_t.reindex(
        [B, N, k, C],
        ["i0", "i1", "i3"],
    )
    feature = jt.concat([neighbors - centers, centers], dim=3)
    return feature.permute(0, 3, 1, 2)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(scale=0.2),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(scale=0.2),
        )

    def execute(self, x):
        x = get_graph_feature(x, self.k)
        x = self.net(x)
        return jt.max(x, dim=3)


class OffsetAttention(nn.Module):
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
        q = self.q_conv(x).permute(0, 2, 1)
        k = self.k_conv(x)
        v = self.v_conv(x)
        attn = self.softmax(jt.nn.bmm(q, k))
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        residual = jt.nn.bmm(v, attn)
        residual = self.act(self.after_norm(self.trans_conv(x - residual)))
        return x + residual


class FullPCT(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        k=20,
        tokens=256,
        width=192,
        emb_dims=1024,
        dropout=0.35,
        use_mean_pool=True,
    ):
        super().__init__()
        self.tokens = tokens
        self.use_mean_pool = use_mean_pool
        self.edge1 = EdgeConvBlock(3, 64, k)
        self.edge2 = EdgeConvBlock(64, 128, k)
        self.proj = nn.Sequential(
            nn.Conv1d(192, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(scale=0.2),
        )

        self.oa1 = OffsetAttention(width)
        self.oa2 = OffsetAttention(width)
        self.oa3 = OffsetAttention(width)
        self.oa4 = OffsetAttention(width)

        self.fuse = nn.Sequential(
            nn.Conv1d(width * 5, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(scale=0.2),
        )

        fc_in = emb_dims * 2 if use_mean_pool else emb_dims
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(scale=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(scale=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def execute(self, x):
        B, C, N = x.shape
        if self.tokens > 0 and N > self.tokens:
            idx = jt.array(np.linspace(0, N - 1, self.tokens, dtype=np.int32))
            x = x.reindex([B, C, self.tokens], ["i0", "i1", "@e0(i2)"], extras=[idx])

        local1 = self.edge1(x)
        local2 = self.edge2(local1)
        x = self.proj(jt.concat([local1, local2], dim=1))

        x1 = self.oa1(x)
        x2 = self.oa2(x1)
        x3 = self.oa3(x2)
        x4 = self.oa4(x3)

        x = self.fuse(jt.concat([x, x1, x2, x3, x4], dim=1))
        x_max = jt.max(x, dim=2)
        if self.use_mean_pool:
            x = jt.concat([x_max, jt.mean(x, dim=2)], dim=1)
        else:
            x = x_max
        return self.classifier(x)


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

        preds = logits.argmax(dim=1)[0]
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

    model = FullPCT(
        num_classes=NUM_CLASSES,
        k=args.k,
        tokens=args.tokens,
        width=args.width,
        emb_dims=args.emb_dims,
        dropout=args.dropout,
        use_mean_pool=args.mean_pool,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Train={train_loader.total_len} Val={val_loader.total_len} Test={test_loader.total_len}")
    print(f"Params={n_params / 1e6:.2f}M CUDA={jt.flags.use_cuda} k={args.k} tokens={args.tokens}")

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
    results = predict(model, test_loader, votes=args.votes)
    write_result(results, args.result_path, args.zip_path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data_dir", default="./data")
    train_parser.add_argument("--n_points", type=int, default=1024)
    train_parser.add_argument("--batch_size", type=int, default=12)
    train_parser.add_argument("--epochs", type=int, default=220)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--min_lr", type=float, default=5e-5)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--val_ratio", type=float, default=0.1)
    train_parser.add_argument("--votes", type=int, default=10)
    train_parser.add_argument("--val_votes", type=int, default=3)
    train_parser.add_argument("--k", type=int, default=20)
    train_parser.add_argument("--tokens", type=int, default=256)
    train_parser.add_argument("--width", type=int, default=192)
    train_parser.add_argument("--emb_dims", type=int, default=1024)
    train_parser.add_argument("--dropout", type=float, default=0.35)
    train_parser.add_argument("--mean_pool", action="store_true", default=True)
    train_parser.add_argument("--no_mean_pool", dest="mean_pool", action="store_false")
    train_parser.add_argument("--label_smoothing", type=float, default=0.1)
    train_parser.add_argument("--test_augment", action="store_true")
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cuda", action="store_true", default=True)
    train_parser.add_argument("--no_cuda", dest="cuda", action="store_false")
    train_parser.add_argument("--checkpoint", default="best_model_full_pct.pkl")
    train_parser.add_argument("--result_path", default="result_full_pct.json")
    train_parser.add_argument("--zip_path", default="result_full_pct.zip")
    train_parser.add_argument("--log_interval", type=int, default=100)

    args = parser.parse_args()
    if args.command is None:
        args.command = "train"
    if args.command != "train":
        raise ValueError(args.command)
    train(args)


if __name__ == "__main__":
    main()
