"""Train a distilled student BPNet model on teacher-generated K562 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "procap" / "processed" / "K562" / "distillation" / "distillation_dataset_k562.npz"

# Add project modules to path lazily inside main to avoid module import at load time.


class DistillationDataset(Dataset):
    """Memory-mapped wrapper around the distillation NPZ archive."""

    def __init__(self, npz_path: Path, indices: np.ndarray) -> None:
        self._path = Path(npz_path)
        self._archive = np.load(self._path, mmap_mode="r")
        self.inputs = self._archive["inputs"]
        self.teacher_log_probs = self._archive["teacher_log_probs"]
        self.teacher_log_counts = self._archive["teacher_log_counts"]
        self.teacher_profile_counts = self._archive["teacher_profile_counts"]
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.indices.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example_id = int(self.indices[idx])
        x = torch.from_numpy(self.inputs[example_id])
        log_probs = torch.from_numpy(self.teacher_log_probs[example_id])
        log_counts = torch.from_numpy(self.teacher_log_counts[example_id])
        profile_counts = torch.from_numpy(self.teacher_profile_counts[example_id])
        return {
            "inputs": x,
            "teacher_log_probs": log_probs,
            "teacher_log_counts": log_counts,
            "teacher_profile_counts": profile_counts,
        }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    count_loss_weight: float,
) -> Tuple[float, float, float]:
    """Return `(total, prob_loss, count_loss)` averaged across batches."""

    model.eval()
    prob_losses, count_losses = [], []

    for batch in data_loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        teacher_log_probs = batch["teacher_log_probs"].to(device, non_blocking=True)
        teacher_log_counts = batch["teacher_log_counts"].to(device, non_blocking=True)

        logits, student_log_counts = model(inputs)
        student_log_probs = model.log_softmax(logits)

        teacher_probs = torch.exp(teacher_log_probs)
        prob_loss = torch.nn.functional.kl_div(
            student_log_probs.view(inputs.size(0), -1),
            teacher_probs.view(inputs.size(0), -1),
            reduction="batchmean",
            log_target=False,
        )
        count_loss = torch.nn.functional.mse_loss(student_log_counts, teacher_log_counts)

        prob_losses.append(prob_loss.detach())
        count_losses.append(count_loss.detach())

    mean_prob = torch.stack(prob_losses).mean().item() if prob_losses else 0.0
    mean_count = torch.stack(count_losses).mean().item() if count_losses else 0.0
    total = mean_prob + count_loss_weight * mean_count
    return total, mean_prob, mean_count


def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    count_loss_weight: float,
) -> Tuple[float, float, float]:
    """Return `(total, prob_loss, count_loss)` averaged across batches."""

    model.train()
    prob_losses, count_losses = [], []

    for batch in data_loader:
        inputs = batch["inputs"].to(device)
        teacher_log_probs = batch["teacher_log_probs"].to(device)
        teacher_log_counts = batch["teacher_log_counts"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, student_log_counts = model(inputs)
        student_log_probs = model.log_softmax(logits)

        teacher_probs = torch.exp(teacher_log_probs)
        prob_loss = torch.nn.functional.kl_div(
            student_log_probs.view(inputs.size(0), -1),
            teacher_probs.view(inputs.size(0), -1),
            reduction="batchmean",
            log_target=False,
        )
        count_loss = torch.nn.functional.mse_loss(student_log_counts, teacher_log_counts)

        loss = prob_loss + count_loss_weight * count_loss
        loss.backward()
        optimizer.step()

        prob_losses.append(prob_loss.detach())
        count_losses.append(count_loss.detach())

    mean_prob = torch.stack(prob_losses).mean().item() if prob_losses else 0.0
    mean_count = torch.stack(count_losses).mean().item() if count_losses else 0.0
    total = mean_prob + count_loss_weight * mean_count
    return total, mean_prob, mean_count


def make_dataloaders(
    archive_path: Path,
    batch_size: int,
    val_fraction: float,
    seed: int,
    train_limit: int | None,
    val_limit: int | None,
    base_only: bool,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    archive = np.load(archive_path, mmap_mode="r")
    n_examples = archive["inputs"].shape[0]
    aug = archive.get("augmentation_id")
    archive.close()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_examples)
    n_val = max(1, int(n_examples * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if train_limit is not None:
        train_idx = train_idx[:train_limit]
    if val_limit is not None:
        val_idx = val_idx[:val_limit]

    # Optionally keep only base (augmentation_id == 0) samples in validation
    if base_only and aug is not None:
        mask = aug[val_idx] == 0
        val_idx = val_idx[mask]

    train_dataset = DistillationDataset(archive_path, train_idx)
    val_dataset = DistillationDataset(archive_path, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, val_idx


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to the distillation NPZ archive.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--count-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on training samples.")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional cap on validation samples.")
    parser.add_argument("--n-filters", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ProCapNet/models/distilled_student"),
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--experiment-npz",
        type=Path,
        default=None,
        help="Optional NPZ with experimental profiles aligned to distillation archive for per-epoch eval.",
    )
    parser.add_argument(
        "--eval-base-only",
        action="store_true",
        help="When evaluating vs experiment, restrict to augmentation_id == 0 (base examples).",
    )
    parser.add_argument(
        "--exp-plot-path",
        type=Path,
        default=None,
        help="If set, save a PNG of student-vs-experiment validation metrics across epochs.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import sys as _sys

    _sys.path.append(str(PROJECT_ROOT / "src" / "2_train_models"))
    _sys.path.append(str(PROJECT_ROOT / "src" / "utils"))
    from BPNet_strand_merged_umap import Model  # type: ignore

    trimming = (2114 - 1000) // 2
    model = Model(
        model_save_path=str(args.output_dir / "student.model"),
        n_filters=args.n_filters,
        n_layers=args.n_layers,
        trimming=trimming,
    ).to(device)

    archive_path = Path(args.archive).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Could not find distillation archive at {archive_path}")

    train_loader, val_loader, val_indices = make_dataloaders(
        archive_path,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        base_only=args.eval_base_only,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"

    history: List[Dict] = []
    best_val = float("inf")
    best_state = None

    # Optionally map experimental profiles once (memory-mapped)
    exp_profiles = None
    exp_totals = None
    if args.experiment_npz is not None:
        exp_path = Path(args.experiment_npz).expanduser().resolve()
        if not exp_path.exists():
            raise FileNotFoundError(f"Experimental NPZ not found: {exp_path}")
        exp_archive = np.load(exp_path, mmap_mode="r")
        exp_profiles = exp_archive["experimental_profile_counts"]
        exp_totals = exp_profiles.sum(axis=(1, 2))

    def _eval_student_vs_experiment() -> Dict[str, float]:
        if exp_profiles is None:
            return {}

        device_local = device
        model.eval()
        log_softmax = torch.nn.LogSoftmax(dim=-1)

        prof_corrs: List[float] = []
        cnt_corrs: List[float] = []
        cnt_mses: List[float] = []

        offset = 0
        for batch in val_loader:
            X = batch["inputs"].to(device_local, non_blocking=True)
            with torch.no_grad():
                logits, log_counts = model(X)
                flat = logits.reshape(logits.shape[0], -1)
                log_probs = log_softmax(flat).reshape_as(logits)
                probs = torch.exp(log_probs)
                total_counts = torch.exp(log_counts) - 1.0
                pred_profiles = probs * total_counts.reshape(-1, 1, 1)

            bsz = pred_profiles.shape[0]
            idx_slice = val_indices[offset : offset + bsz]
            offset += bsz

            true_profiles = exp_profiles[idx_slice]

            # Move predictions to CPU numpy
            pred_np = pred_profiles.detach().cpu().numpy().astype(np.float32, copy=False)

            # Per-example metrics
            for i in range(bsz):
                p = pred_np[i]
                t = true_profiles[i]

                # Profile Pearson (flattened across strands and positions)
                x = p.reshape(-1)
                y = t.reshape(-1)
                if np.std(x) == 0 or np.std(y) == 0:
                    prof_corr = np.nan
                else:
                    prof_corr = np.corrcoef(x, y)[0, 1]
                prof_corrs.append(float(prof_corr))

                # Count Pearson across positions (sum across strands)
                px = p.sum(axis=0)
                py = t.sum(axis=0)
                if np.std(px) == 0 or np.std(py) == 0:
                    cnt_corr = np.nan
                else:
                    cnt_corr = np.corrcoef(px, py)[0, 1]
                cnt_corrs.append(float(cnt_corr))

                # Count log1pMSE on scalar totals
                tot_p = p.sum()
                tot_t = t.sum()
                cnt_mse = float((np.log1p(tot_p) - np.log1p(tot_t)) ** 2)
                cnt_mses.append(cnt_mse)

        return {
            "exp_profile_pearson_mean": float(np.nanmean(prof_corrs)) if prof_corrs else float("nan"),
            "exp_profile_pearson_median": float(np.nanmedian(prof_corrs)) if prof_corrs else float("nan"),
            "exp_count_pearson_mean": float(np.nanmean(cnt_corrs)) if cnt_corrs else float("nan"),
            "exp_count_pearson_median": float(np.nanmedian(cnt_corrs)) if cnt_corrs else float("nan"),
            "exp_count_log1pMSE_mean": float(np.nanmean(cnt_mses)) if cnt_mses else float("nan"),
            "exp_count_log1pMSE_median": float(np.nanmedian(cnt_mses)) if cnt_mses else float("nan"),
        }

    for epoch in range(1, args.epochs + 1):
        train_total, train_prob, train_count = train_epoch(
            model, train_loader, optimizer, device, args.count_loss_weight
        )
        val_total, val_prob, val_count = evaluate(
            model, val_loader, device, args.count_loss_weight
        )
        record: Dict = {
            "epoch": epoch,
            "train_total": train_total,
            "train_prob": train_prob,
            "train_count": train_count,
            "val_total": val_total,
            "val_prob": val_prob,
            "val_count": val_count,
        }

        # Optionally evaluate vs experiment for this epoch
        if exp_profiles is not None:
            exp_metrics = _eval_student_vs_experiment()
            record.update(exp_metrics)
        history.append(record)
        print(
            f"Epoch {epoch:02d}: train_total={train_total:.4f} "
            f"(prob={train_prob:.4f}, count={train_count:.4f}) | "
            f"val_total={val_total:.4f} (prob={val_prob:.4f}, count={val_count:.4f})"
        )
        if val_total < best_val:
            best_val = val_total
            best_state = {"epoch": epoch, "model": model.state_dict()}

    with metrics_path.open("w") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")

    if best_state is not None:
        torch.save(best_state, args.output_dir / "student_best.pt")
    torch.save(model.state_dict(), args.output_dir / "student_last.pt")

    # Optional plotting of student-vs-experiment metrics across epochs
    if args.exp_plot_path is not None and args.experiment_npz is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping student-vs-experiment plot")
        else:
            exp_plot = Path(args.exp_plot_path).expanduser().resolve()
            exp_plot.parent.mkdir(parents=True, exist_ok=True)

            epochs = [h["epoch"] for h in history]
            def series(key: str):
                return [h.get(key, np.nan) for h in history]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, series("exp_profile_pearson_mean"), label="Profile Pearson (mean)")
            plt.plot(epochs, series("exp_profile_pearson_median"), label="Profile Pearson (median)")
            plt.plot(epochs, series("exp_count_pearson_mean"), label="Count Pearson (mean)")
            plt.plot(epochs, series("exp_count_pearson_median"), label="Count Pearson (median)")
            plt.xlabel("Epoch")
            plt.ylabel("Correlation")
            plt.title("Student vs Experiment validation metrics")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(exp_plot)
            plt.close()
            print(f"Saved student-vs-experiment metrics plot to {exp_plot}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
