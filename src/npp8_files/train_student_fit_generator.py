"""Train a BPNet student via Model.fit_generator using teacher log probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "procap" / "processed" / "K562" / "distillation" / "distillation_dataset_k562.npz"
DEFAULT_SAVE = PROJECT_ROOT / "models" / "distilled_fit_generator" / "student.model"


class FitGeneratorDataset(Dataset):
    """Wrap the distillation NPZ with on-the-fly target reconstruction."""

    def __init__(self, archive_path: Path, indices: np.ndarray) -> None:
        self._path = Path(archive_path)
        self._archive = np.load(self._path, mmap_mode="r")
        self.inputs = self._archive["inputs"]
        self.teacher_log_probs = self._archive["teacher_log_probs"]
        self.teacher_log_counts = self._archive["teacher_log_counts"]
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return self.indices.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example_idx = int(self.indices[idx])

        inputs = torch.from_numpy(self.inputs[example_idx])
        log_probs = torch.from_numpy(self.teacher_log_probs[example_idx])
        log_count = torch.from_numpy(self.teacher_log_counts[example_idx])

        probs = torch.exp(log_probs)
        total_count = torch.exp(log_count).item()
        targets = probs * total_count

        mask = torch.ones_like(targets, dtype=torch.bool)
        return inputs, targets, mask


def _split_indices(n_examples: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive)")

    rng = np.random.default_rng(seed)
    ordering = rng.permutation(n_examples)
    n_val = max(1, int(n_examples * val_fraction))
    if n_val >= n_examples:
        n_val = n_examples - 1
    val_idx = ordering[:n_val]
    train_idx = ordering[n_val:]
    return train_idx, val_idx


def _load_validation_arrays(archive_path: Path, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(archive_path, mmap_mode="r") as archive:
        inputs = np.ascontiguousarray(archive["inputs"][indices], dtype=np.float32)
        log_probs = archive["teacher_log_probs"][indices]
        log_counts = archive["teacher_log_counts"][indices]

    probs = np.empty_like(log_probs)
    np.exp(log_probs, out=probs)

    total_counts = np.empty_like(log_counts)
    np.exp(log_counts, out=total_counts)

    targets = probs * total_counts.reshape(-1, 1, 1)
    return inputs, targets.astype(np.float32, copy=False)


def make_training_loader(
    archive_path: Path,
    batch_size: int,
    val_fraction: float,
    seed: int,
    train_limit: int | None,
    val_limit: int | None,
) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    with np.load(archive_path, mmap_mode="r") as archive:
        n_examples = archive["inputs"].shape[0]

    train_idx, val_idx = _split_indices(n_examples, val_fraction, seed)

    if train_limit is not None:
        train_idx = train_idx[:train_limit]
    if val_limit is not None:
        val_idx = val_idx[:val_limit]

    if train_idx.size == 0:
        raise ValueError("Training split is empty; adjust train_limit or val_fraction")

    train_dataset = FitGeneratorDataset(archive_path, train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    X_valid, y_valid = _load_validation_arrays(archive_path, val_idx)
    return train_loader, X_valid, y_valid


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="Path to the distillation NPZ archive.")
    parser.add_argument("--model-save-path", type=Path, default=DEFAULT_SAVE, help="Destination for the trained model checkpoint.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-iter", type=int, default=200, help="Validate every N training iterations.")
    parser.add_argument("--early-stop-epochs", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for the count loss term.")
    parser.add_argument("--n-filters", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None, help="Optionally cap training samples for quick tests.")
    parser.add_argument("--val-limit", type=int, default=None, help="Optionally cap validation samples for quick tests.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose training logs.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    archive_path = args.archive.expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Could not find archive at {archive_path}")

    # Delay heavy imports until paths are configured.
    import sys as _sys

    _sys.path.append(str(PROJECT_ROOT / "src" / "2_train_models"))
    _sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

    from BPNet_strand_merged_umap import Model  # type: ignore
    from misc import ensure_parent_dir_exists  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trimming = (2114 - 1000) // 2

    model = Model(
        model_save_path=str(args.model_save_path),
        n_filters=args.n_filters,
        n_layers=args.n_layers,
        n_outputs=2,
        alpha=args.alpha,
        trimming=trimming,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, X_valid, y_valid = make_training_loader(
        archive_path=archive_path,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )

    ensure_parent_dir_exists(str(args.model_save_path))
    if not args.quiet:
        arch_txt = args.model_save_path.with_suffix(".arch.txt")
        model.save_model_arch_to_txt(str(arch_txt))

    model.fit_generator(
        training_data=train_loader,
        optimizer=optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        validation_iter=args.validation_iter,
        early_stop_epochs=args.early_stop_epochs,
        verbose=not args.quiet,
        save=True,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
