#!/usr/bin/env python3
"""
Run the distilled student BPNet on the distillation archive and save predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InputsDataset(Dataset):
    """Wrap the NPZ inputs for batch-wise inference."""

    def __init__(self, npz_path: Path) -> None:
        self._path = Path(npz_path)
        archive = np.load(self._path, mmap_mode="r")
        self.inputs = archive["inputs"]
        self.origin_index = archive.get("origin_index")
        self.augmentation_id = archive.get("augmentation_id")
        archive.close()

    def __len__(self) -> int:  # type: ignore[override]
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(self.inputs[idx])
        item = {"inputs": x}
        if self.origin_index is not None:
            item["origin_index"] = torch.tensor(self.origin_index[idx], dtype=torch.long)
        if self.augmentation_id is not None:
            item["augmentation_id"] = torch.tensor(self.augmentation_id[idx], dtype=torch.long)
        return item


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to distillation NPZ (must contain `inputs`).",
    )
    parser.add_argument(
        "--model-state",
        type=Path,
        required=True,
        help="Path to the student model checkpoint (`student.model`).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination NPZ to store student predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    archive_path = args.archive.expanduser().resolve(strict=True)
    model_path = args.model_state.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import (so path setup happens first)
    import sys as _sys
    _sys.path.append(str(PROJECT_ROOT / "src" / "2_train_models"))
    _sys.path.append(str(PROJECT_ROOT / "src" / "utils"))
    from BPNet_strand_merged_umap import Model  # type: ignore

    device = torch.device(args.device)

    print(f"Loading student model from {model_path}")
    from torch.serialization import safe_globals

    with safe_globals([Model]):
        model: Model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    dataset = InputsDataset(archive_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    log_softmax = torch.nn.LogSoftmax(dim=-1)

    all_log_probs: list[np.ndarray] = []
    all_log_counts: list[np.ndarray] = []
    all_profile_counts: list[np.ndarray] = []
    origin_indices: list[np.ndarray] = []
    augmentation_ids: list[np.ndarray] = []

    print("Running inference on student model...")
    with torch.no_grad():
        for batch in loader:
            X = batch["inputs"].to(device, non_blocking=True)
            logits, log_counts = model(X)

            flat = logits.reshape(logits.shape[0], -1)
            log_probs = log_softmax(flat).reshape_as(logits)

            probs = torch.exp(log_probs)
            total_counts = torch.exp(log_counts) - 1.0
            profile_counts = probs * total_counts.reshape(-1, 1, 1)

            all_log_probs.append(log_probs.cpu().numpy())
            all_log_counts.append(log_counts.cpu().numpy())
            all_profile_counts.append(profile_counts.cpu().numpy())

            if "origin_index" in batch:
                origin_indices.append(batch["origin_index"].cpu().numpy())
            if "augmentation_id" in batch:
                augmentation_ids.append(batch["augmentation_id"].cpu().numpy())

    student_log_probs = np.concatenate(all_log_probs, axis=0).astype(np.float32, copy=False)
    student_log_counts = np.concatenate(all_log_counts, axis=0).astype(np.float32, copy=False)
    student_profile_counts = np.concatenate(all_profile_counts, axis=0).astype(np.float32, copy=False)

    save_kwargs = {
        "student_log_probs": student_log_probs,
        "student_log_counts": student_log_counts,
        "student_profile_counts": student_profile_counts,
    }
    if origin_indices:
        save_kwargs["origin_index"] = np.concatenate(origin_indices, axis=0).astype(np.int64, copy=False)
    if augmentation_ids:
        save_kwargs["augmentation_id"] = np.concatenate(augmentation_ids, axis=0).astype(np.int16, copy=False)

    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved student predictions to {output_path}")


if __name__ == "__main__":
    main()
