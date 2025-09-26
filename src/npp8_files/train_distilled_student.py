"""Train a distilled student BPNet model on teacher-generated K562 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
) -> Tuple[DataLoader, DataLoader]:
    archive = np.load(archive_path, mmap_mode="r")
    n_examples = archive["inputs"].shape[0]
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
    return train_loader, val_loader


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
    parser.add_argument("--n-filters", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ProCapNet/models/distilled_student"),
        help="Directory to store checkpoints and logs.",
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

    train_loader, val_loader = make_dataloaders(
        archive_path,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.jsonl"

    history = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_total, train_prob, train_count = train_epoch(
            model, train_loader, optimizer, device, args.count_loss_weight
        )
        val_total, val_prob, val_count = evaluate(
            model, val_loader, device, args.count_loss_weight
        )
        record = {
            "epoch": epoch,
            "train_total": train_total,
            "train_prob": train_prob,
            "train_count": train_count,
            "val_total": val_total,
            "val_prob": val_prob,
            "val_count": val_count,
        }
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


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
