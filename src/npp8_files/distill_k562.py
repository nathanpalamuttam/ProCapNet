"""Utilities to generate teacher labels and augmented inputs for K562 distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

proj_root = Path(__file__).resolve().parents[2]
train_src = proj_root / "src" / "2_train_models"
utils_src = proj_root / "src" / "utils"

import sys
if str(train_src) not in sys.path:
    sys.path.append(str(train_src))
if str(utils_src) not in sys.path:
    sys.path.append(str(utils_src))

from BPNet_strand_merged_umap import Model  # type: ignore  # noqa: E402
from misc import ensure_parent_dir_exists  # type: ignore  # noqa: E402

from gen_one_hot_encoding import augment_onehot_sequences  # type: ignore  # noqa: E402


DEFAULT_TIMESTAMPS = (
    "2023-05-29_15-51-40",
    "2023-05-29_15-58-41",
    "2023-05-29_15-59-09",
    "2023-05-30_01-40-06",
    "2023-05-29_23-21-23",
    "2023-05-29_23-23-45",
    "2023-05-29_23-24-11",
)


def _load_teacher_models(timestamps: Sequence[str], cell_type: str, device: torch.device) -> List[Model]:
    models: List[Model] = []
    model_dir = proj_root / "models" / "procap" / cell_type / "strand_merged_umap"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    for ts in timestamps:
        model_path = model_dir / f"{ts}.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing teacher checkpoint: {model_path}")

        model: Model = torch.load(model_path, map_location=device)
        model = model.to(device)
        model.eval()
        models.append(model)

    return models


def _predict_ensemble(
    models: Sequence[Model],
    inputs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ensemble-averaged log probs, log counts, and profile counts."""

    if inputs.dtype != np.float32:
        inputs = inputs.astype(np.float32, copy=False)

    n_examples = inputs.shape[0]
    n_strands = models[0].n_outputs
    out_window = models[0].trimming * -2 + inputs.shape[-1]

    log_prob_out = np.empty((n_examples, n_strands, out_window), dtype=np.float32)
    log_count_out = np.empty((n_examples, 1), dtype=np.float32)
    profile_counts_out = np.empty((n_examples, n_strands, out_window), dtype=np.float32)

    log_softmax = torch.nn.LogSoftmax(dim=-1)

    for start in range(0, n_examples, batch_size):
        end = min(start + batch_size, n_examples)
        batch = torch.from_numpy(inputs[start:end]).to(device)

        prob_accum = None
        logcount_stack = []

        for model in models:
            with torch.no_grad():
                y_profile, y_counts = model(batch)
                flat = y_profile.reshape(y_profile.shape[0], -1)
                log_probs = log_softmax(flat).reshape_as(y_profile)
            log_probs_np = log_probs.detach().cpu().numpy()
            log_counts_np = y_counts.detach().cpu().numpy()
            probs_np = np.exp(log_probs_np).astype(np.float32, copy=False)

            if prob_accum is None:
                prob_accum = probs_np
            else:
                prob_accum += probs_np

            logcount_stack.append(log_counts_np)

        prob_avg = prob_accum / float(len(models))
        logcount_avg = np.mean(np.stack(logcount_stack, axis=0), axis=0)
        total_counts = np.exp(logcount_avg).astype(np.float32, copy=False)
        profile_counts = prob_avg * total_counts.reshape(-1, 1, 1)

        log_prob_out[start:end] = np.log(np.clip(prob_avg, 1e-12, None)).astype(np.float32, copy=False)
        log_count_out[start:end] = logcount_avg.astype(np.float32, copy=False)
        profile_counts_out[start:end] = profile_counts.astype(np.float32, copy=False)

    return log_prob_out, log_count_out, profile_counts_out


def _stack_batches(batches: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(batches)
    if not arrays:
        raise ValueError("No arrays to stack")
    return np.concatenate(arrays, axis=0)


def _write_metadata(metadata_path: Path, metadata: dict) -> None:
    ensure_parent_dir_exists(str(metadata_path))
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)


def run_pipeline(
    cell_type: str = "K562",
    timestamps: Sequence[str] = DEFAULT_TIMESTAMPS,
    batch_size: int = 128,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_augmentations: int = 1,
    shift_range: int = 1024,
    rc_prob: float = 0.5,
    base_seed: int = 123,
    output_dir: Path | None = None,
) -> None:
    device = torch.device(device_str)

    sequence_path = proj_root / "data" / "procap" / "processed" / cell_type / "onehot_seqs_2114.npy"
    if not sequence_path.exists():
        raise FileNotFoundError(f"Missing one-hot training sequences at {sequence_path}")

    print(f"Loading base sequences from {sequence_path}")
    base_sequences = np.load(sequence_path)
    if base_sequences.dtype != np.float32:
        base_sequences = base_sequences.astype(np.float32, copy=False)

    models = _load_teacher_models(timestamps, cell_type, device)

    print("Generating teacher predictions for base sequences...")
    base_log_probs, base_log_counts, base_profile_counts = _predict_ensemble(
        models,
        base_sequences,
        device,
        batch_size,
    )

    all_inputs = [base_sequences]
    all_log_probs = [base_log_probs]
    all_log_counts = [base_log_counts]
    all_profile_counts = [base_profile_counts]
    augment_indices = [np.arange(base_sequences.shape[0], dtype=np.int64)]
    augment_ids = [np.zeros(base_sequences.shape[0], dtype=np.int16)]

    for aug_idx in range(1, num_augmentations + 1):
        seed = base_seed + aug_idx
        print(f"Augmenting sequences (augmentation #{aug_idx}, seed={seed})")
        aug_sequences, _ = augment_onehot_sequences(
            base_sequences,
            outputs=None,
            shift_range=shift_range,
            rc_prob=rc_prob,
            seed=seed,
        )
        aug_sequences = aug_sequences.astype(np.float32, copy=False)
        aug_inputs = aug_sequences

        print("Generating teacher predictions for augmented sequences...")
        aug_log_probs, aug_log_counts, aug_profile_counts = _predict_ensemble(
            models,
            aug_inputs,
            device,
            batch_size,
        )

        all_inputs.append(aug_inputs)
        all_log_probs.append(aug_log_probs)
        all_log_counts.append(aug_log_counts)
        all_profile_counts.append(aug_profile_counts)

        augment_indices.append(np.arange(base_sequences.shape[0], dtype=np.int64))
        augment_ids.append(np.full(base_sequences.shape[0], aug_idx, dtype=np.int16))

    inputs_full = _stack_batches(all_inputs)
    log_probs_full = _stack_batches(all_log_probs)
    log_counts_full = _stack_batches(all_log_counts)
    profile_counts_full = _stack_batches(all_profile_counts)
    origin_indices = _stack_batches(augment_indices)
    augmentation_ids = _stack_batches(augment_ids)

    if output_dir is None:
        output_dir = proj_root / "data" / "procap" / "processed" / cell_type / "distillation"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "distillation_dataset_k562.npz"
    print(f"Writing aggregated dataset to {dataset_path}")
    np.savez_compressed(
        dataset_path,
        inputs=inputs_full.astype(np.float32),
        teacher_log_probs=log_probs_full,
        teacher_log_counts=log_counts_full,
        teacher_profile_counts=profile_counts_full,
        origin_index=origin_indices,
        augmentation_id=augmentation_ids,
    )

    metadata = {
        "cell_type": cell_type,
        "timestamps": list(timestamps),
        "batch_size": batch_size,
        "device": device_str,
        "num_augmentations": num_augmentations,
        "shift_range": shift_range,
        "rc_prob": rc_prob,
        "base_seed": base_seed,
        "sequence_path": str(sequence_path),
        "dataset_path": str(dataset_path),
        "n_examples_original": int(base_sequences.shape[0]),
        "n_examples_total": int(inputs_full.shape[0]),
    }

    metadata_path = output_dir / "distillation_dataset_k562.json"
    _write_metadata(metadata_path, metadata)
    print(f"Saved metadata to {metadata_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare K562 distillation dataset")
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--timestamps", nargs="*", default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="torch device string")
    parser.add_argument("--num-augmentations", type=int, default=1)
    parser.add_argument("--shift-range", type=int, default=1024)
    parser.add_argument("--rc-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_pipeline(
        cell_type=args.cell_type,
        timestamps=args.timestamps,
        batch_size=args.batch_size,
        device_str=device_str,
        num_augmentations=max(0, args.num_augmentations),
        shift_range=args.shift_range,
        rc_prob=args.rc_prob,
        base_seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
