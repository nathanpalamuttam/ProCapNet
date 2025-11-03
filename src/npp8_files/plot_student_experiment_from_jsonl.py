#!/usr/bin/env python3
"""Plot Student vs Experiment validation metrics across epochs.

Reads the JSONL produced by train_distilled_student.py (when run with
--experiment-npz) and plots per-epoch correlations/errors between the
student predictions and experimental tracks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-jsonl",
        type=Path,
        required=True,
        help="Path to training_metrics.jsonl from train_distilled_student.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination PNG for the plot",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling window over epochs (>=1; 1 disables)",
    )
    p.add_argument(
        "--which",
        choices=["mean", "median", "both"],
        default="both",
        help="Plot mean, median, or both curves",
    )
    return p.parse_args(argv)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    w = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, w, mode="valid")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    rows = load_jsonl(args.metrics_jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.metrics_jsonl}")

    epochs = np.array([r.get("epoch", i + 1) for i, r in enumerate(rows)], dtype=int)

    keys = {
        "profile_mean": "exp_profile_pearson_mean",
        "profile_median": "exp_profile_pearson_median",
        "count_mean": "exp_count_pearson_mean",
        "count_median": "exp_count_pearson_median",
        "mse_mean": "exp_count_log1pMSE_mean",
        "mse_median": "exp_count_log1pMSE_median",
    }

    data: Dict[str, np.ndarray] = {}
    for name, key in keys.items():
        vals = [r.get(key, np.nan) for r in rows]
        data[name] = np.array(vals, dtype=float)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install it via pip") from exc

    smooth = max(1, int(args.smooth))
    if smooth > 1 and len(epochs) >= smooth:
        x = epochs[smooth - 1 :]
    else:
        x = epochs

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    def plot_pair(ax, mean_key: str, median_key: str, ylabel: str):
        if args.which in ("mean", "both"):
            y = data[mean_key]
            if smooth > 1:
                y = _rolling_mean(y, smooth)
            ax.plot(x, y, label="mean", linewidth=2)
        if args.which in ("median", "both"):
            y = data[median_key]
            if smooth > 1:
                y = _rolling_mean(y, smooth)
            ax.plot(x, y, label="median", linestyle="--")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(loc="best")

    plot_pair(axes[0], "profile_mean", "profile_median", "Profile Pearson")
    plot_pair(axes[1], "count_mean", "count_median", "Count Pearson")
    plot_pair(axes[2], "mse_mean", "mse_median", "Count log1pMSE")
    axes[2].set_xlabel("Epoch")

    ttl = "Student vs Experiment validation metrics"
    if smooth > 1:
        ttl += f" (rolling={smooth})"
    fig.suptitle(ttl)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()

