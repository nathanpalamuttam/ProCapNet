#!/usr/bin/env python3
"""Plot validation metrics vs. epoch from the fit_generator TSV log.

Reads a TSV like models/distilled_fit_generator/student_metrics.tsv and
aggregates per-iteration validation metrics by epoch, then saves a summary plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-tsv", type=Path, required=True, help="Path to student_metrics.tsv")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path for the plot")
    p.add_argument(
        "--agg",
        choices=["mean", "last"],
        default="mean",
        help="Aggregate per-epoch metric across iterations (mean or last)",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Optional rolling window over epochs (>=1; 1 disables)",
    )
    p.add_argument(
        "--loss-out",
        type=Path,
        default=None,
        help="Optional second PNG path for training/validation loss curves.",
    )
    p.add_argument(
        "--loss-separate-out",
        type=Path,
        default=None,
        help="Optional PNG path to save a 3-panel figure: MNLL, Count log1pMSE, and Total loss separately.",
    )
    p.add_argument(
        "--ylog-loss",
        action="store_true",
        help="Use log scale on the y-axis for loss plots (helps when MNLL dwarfs count loss).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight applied to Count log1pMSE when forming total loss = MNLL + alpha * Count log1pMSE",
    )
    return p.parse_args(argv)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(y):
        return y
    w = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, w, mode="valid")


def load_and_aggregate(tsv_path: Path, agg: str) -> Dict[str, np.ndarray]:
    # Collect metrics per epoch
    per_epoch: Dict[int, Dict[str, List[float]]] = {}

    with tsv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = [
            "Epoch",
            "Val MNLL",
            "Val JSD",
            "Val Profile Pearson",
            "Val Count Pearson",
            "Val Count log1pMSE",
        ]
        for req in required:
            if req not in reader.fieldnames:
                raise ValueError(f"Column '{req}' not found. Available: {reader.fieldnames}")

        # Training columns are optional (older logs may omit)
        train_cols = ["Train MNLL", "Train Count log1pMSE"]
        have_train = all(col in reader.fieldnames for col in train_cols)

        for row in reader:
            try:
                epoch = int(row["Epoch"])
            except Exception:
                continue
            keys = [k for k in required if k != "Epoch"]
            if have_train:
                keys += train_cols
            bucket = per_epoch.setdefault(epoch, {k: [] for k in keys})
            for k in bucket.keys():
                v = row.get(k, "")
                if v in ("", "nan", "NaN", "NA"):
                    continue
                try:
                    bucket[k].append(float(v))
                except Exception:
                    pass

    # Aggregate per epoch
    epochs = np.array(sorted(per_epoch.keys()))
    out: Dict[str, np.ndarray] = {"epoch": epochs}
    for metric in [
        "Val MNLL",
        "Val JSD",
        "Val Profile Pearson",
        "Val Count Pearson",
        "Val Count log1pMSE",
        "Train MNLL",
        "Train Count log1pMSE",
    ]:
        vals: List[float] = []
        for e in epochs:
            if metric not in per_epoch[e]:
                vals.append(np.nan)
                continue
            series = per_epoch[e][metric]
            if not series:
                vals.append(np.nan)
                continue
            if agg == "mean":
                vals.append(float(np.mean(series)))
            else:
                vals.append(float(series[-1]))
        out[metric] = np.array(vals, dtype=float)
    return out


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_and_aggregate(args.metrics_tsv, args.agg)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install it via pip.") from exc

    x = data["epoch"]

    # Optionally smooth across epochs
    series_names = [
        "Val Profile Pearson",
        "Val Count Pearson",
        "Val JSD",
        "Val MNLL",
        "Val Count log1pMSE",
    ]

    # Apply smoothing; keep aligned x by trimming to valid range
    smooth = max(1, int(args.smooth))
    if smooth > 1 and len(x) >= smooth:
        x_plot = x[smooth - 1 :]
    else:
        x_plot = x

    plt.figure(figsize=(12, 8))
    for name in series_names:
        y = data[name]
        if smooth > 1 and len(y) >= smooth:
            y = _rolling_mean(y, smooth)
        plt.plot(x_plot, y, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    title = "Validation metrics vs epoch"
    if args.agg == "mean":
        title += " (per-epoch mean)"
    else:
        title += " (last validation per epoch)"
    if smooth > 1:
        title += f", rolling={smooth}"
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

    # Optional loss plot (train/val)
    if args.loss_out is not None:
        try:
            import matplotlib.pyplot as plt  # already imported above; keep for clarity
        except ImportError:
            pass
        else:
            train_mnll = data.get("Train MNLL", np.full_like(x, np.nan, dtype=float))
            train_cnt = data.get("Train Count log1pMSE", np.full_like(x, np.nan, dtype=float))
            val_mnll = data.get("Val MNLL")
            val_cnt = data.get("Val Count log1pMSE")

            train_total = train_mnll + args.alpha * train_cnt
            val_total = val_mnll + args.alpha * val_cnt

            def smooth_series(y: np.ndarray) -> np.ndarray:
                if smooth > 1 and len(y) >= smooth:
                    return _rolling_mean(y, smooth)
                return y

            plt.figure(figsize=(12, 6))
            plt.plot(x_plot, smooth_series(train_mnll), label="Train MNLL", alpha=0.8)
            plt.plot(x_plot, smooth_series(train_cnt), label="Train Count log1pMSE", alpha=0.8)
            plt.plot(x_plot, smooth_series(train_total), label=f"Train Total (alpha={args.alpha})", linewidth=2)
            plt.plot(x_plot, smooth_series(val_mnll), label="Val MNLL", linestyle="--", alpha=0.9)
            plt.plot(x_plot, smooth_series(val_cnt), label="Val Count log1pMSE", linestyle="--", alpha=0.9)
            plt.plot(x_plot, smooth_series(val_total), label=f"Val Total (alpha={args.alpha})", linestyle="-.", linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            ttl = "Training/Validation loss vs epoch"
            if smooth > 1:
                ttl += f" (rolling={smooth})"
            plt.title(ttl)
            if args.ylog_loss:
                plt.yscale('log')
            plt.grid(True, linestyle=':', linewidth=0.5)
            plt.legend(loc="best")
            plt.tight_layout()
            out2 = args.loss_out.expanduser().resolve()
            out2.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out2)
            plt.close()
            print(f"Saved loss plot to {out2}")

    # Optional separate 3-panel loss figure
    if args.loss_separate_out is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            train_mnll = data.get("Train MNLL", np.full_like(x, np.nan, dtype=float))
            train_cnt = data.get("Train Count log1pMSE", np.full_like(x, np.nan, dtype=float))
            val_mnll = data.get("Val MNLL")
            val_cnt = data.get("Val Count log1pMSE")

            train_total = train_mnll + args.alpha * train_cnt
            val_total = val_mnll + args.alpha * val_cnt

            def smooth_series(y: np.ndarray) -> np.ndarray:
                if smooth > 1 and len(y) >= smooth:
                    return _rolling_mean(y, smooth)
                return y

            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
            # Panel 1: MNLL
            axes[0].plot(x_plot, smooth_series(train_mnll), label="Train MNLL", alpha=0.85)
            axes[0].plot(x_plot, smooth_series(val_mnll), label="Val MNLL", linestyle='--', alpha=0.9)
            axes[0].set_ylabel('MNLL')
            axes[0].grid(True, linestyle=':', linewidth=0.5)
            axes[0].legend(loc='best')
            if args.ylog_loss:
                axes[0].set_yscale('log')

            # Panel 2: Count log1pMSE
            axes[1].plot(x_plot, smooth_series(train_cnt), label="Train Count log1pMSE", alpha=0.85)
            axes[1].plot(x_plot, smooth_series(val_cnt), label="Val Count log1pMSE", linestyle='--', alpha=0.9)
            axes[1].set_ylabel('Count log1pMSE')
            axes[1].grid(True, linestyle=':', linewidth=0.5)
            axes[1].legend(loc='best')
            if args.ylog_loss:
                axes[1].set_yscale('log')

            # Panel 3: Total
            axes[2].plot(x_plot, smooth_series(train_total), label=f"Train Total (alpha={args.alpha})", linewidth=2)
            axes[2].plot(x_plot, smooth_series(val_total), label=f"Val Total (alpha={args.alpha})", linestyle='-.', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Total loss')
            axes[2].grid(True, linestyle=':', linewidth=0.5)
            axes[2].legend(loc='best')
            if args.ylog_loss:
                axes[2].set_yscale('log')

            ttl = 'Training/Validation loss by component'
            if smooth > 1:
                ttl += f' (rolling={smooth})'
            fig.suptitle(ttl)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            out3 = args.loss_separate_out.expanduser().resolve()
            out3.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out3)
            plt.close(fig)
            print(f"Saved separate loss panels to {out3}")


if __name__ == "__main__":
    main()
