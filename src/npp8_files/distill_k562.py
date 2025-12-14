"""Utilities for K562 distillation.

Streaming mode that trains a student directly from DistillerPeakGenerator
batches while matching a teacher ensemble, using Model.fit_generator for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple, Optional

import numpy as np
import torch

proj_root = Path(__file__).resolve().parents[2]
train_src = proj_root / "src" / "2_train_models"
utils_src = proj_root / "src" / "utils"
npp8_src = proj_root / "src" / "npp8_files"

import sys
if str(train_src) not in sys.path:
    sys.path.insert(0, str(train_src))
for p in (utils_src, npp8_src):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from BPNet_strand_merged_umap import Model  # type: ignore  # noqa: E402
from data_loader import DistillerPeakGenerator  # type: ignore  # noqa: E402
from misc import ensure_parent_dir_exists  # type: ignore  # noqa: E402


DEFAULT_TIMESTAMPS = (
    "2023-05-29_15-51-40",
    "2023-05-29_15-58-41",
    "2023-05-29_15-59-09",
    "2023-05-30_01-40-06",
    "2023-05-29_23-21-23",
    "2023-05-29_23-23-45",
    "2023-05-29_23-24-11",
)


def _load_teacher_models(timestamps: Sequence[str], cell_type: str) -> List[Model]:
    """Load teacher models and keep them on CPU to save GPU memory."""
    models: List[Model] = []
    model_dir = proj_root / "models" / "procap" / cell_type / "strand_merged_umap"
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    for ts in timestamps:
        model_path = model_dir / f"{ts}.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing teacher checkpoint: {model_path}")

        model: Model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
        models.append(model)  # Keep on CPU

    return models


def _make_streaming_loader(
    cell_type: str,
    in_window: int = 2114,
    out_window: int = 1000,
    batch_size: int = 64,
    negative_ratio: float = 0.1,
    max_jitter: int = 0,
    reverse_complement: bool = False,
    mutation_rate: float = 0.0,
    sv_rate: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
):
    genome_path = proj_root / "genomes" / "hg38.withrDNA.fasta"
    peaks = proj_root / "data" / "procap" / "processed" / cell_type / "peaks.bed.gz"
    neg = proj_root / "data" / "procap" / "processed" / cell_type / "dnase_peaks_no_procap_overlap.bed.gz"
    if not peaks.exists() or not neg.exists() or not genome_path.exists():
        missing = [p for p in (genome_path, peaks, neg) if not p.exists()]
        raise FileNotFoundError(f"Missing required inputs for streaming loader: {missing}")

    loader = DistillerPeakGenerator(
        peaks=str(peaks),
        negatives=str(neg),
        sequences=str(genome_path),
        controls=None,
        chroms=None,
        in_window=in_window,
        out_window=out_window,
        max_jitter=max_jitter,
        negative_ratio=negative_ratio,
        reverse_complement=reverse_complement,
        mutation_rate=mutation_rate,
        sv_rate=sv_rate,
        shuffle=True,
        random_state=seed,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        batch_size=batch_size,
        verbose=verbose,
    )
    return loader


@torch.no_grad()
def _teacher_batch(
    models: Sequence[Model], X: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (teacher_profile_counts, teacher_log_counts) as torch tensors on device.
    
    Teachers are moved to GPU one at a time to save memory.
    """
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    track_accum: Optional[torch.Tensor] = None

    for m in models:
        # Move teacher to GPU for inference
        m = m.to(device)
        m.eval()
        
        logits, log_counts = m(X)
        flat = logits.reshape(logits.shape[0], -1)
        log_probs = log_softmax(flat).reshape_as(logits)
        probs = torch.exp(log_probs)
        total = torch.exp(log_counts) - 1.0

        # Track = probabilities * total counts per example
        track = probs * total.view(-1, 1, 1)

        if track_accum is None:
            track_accum = track.clone()
        else:
            track_accum = track_accum + track

        # Move teacher back to CPU and free GPU memory
        m.cpu()
        del logits, log_counts, flat, log_probs, probs, total, track
        torch.cuda.empty_cache()

    assert track_accum is not None

    # Average tracks directly (E[P*C]), then derive totals and log counts
    track_avg = track_accum / float(len(models))
    total_avg = track_avg.reshape(track_avg.shape[0], -1).sum(dim=1, keepdim=True) + 1.0

    teacher_log_counts = torch.log(torch.clamp(total_avg, min=1e-12))
    return track_avg, teacher_log_counts


class TeacherDistillationGenerator:
    """Wrap a base loader to attach teacher targets and masks for fit_generator.
    
    Yields (X, y, mask) tuples where:
    - X: input sequences (batch_size, 4, in_window)
    - y: teacher profile counts (batch_size, n_outputs, out_window)
    - mask: boolean mask for valid positions (batch_size, n_outputs, out_window)
    
    This matches the format expected by Model.fit_generator from BPNet_strand_merged_umap.
    """

    def __init__(self, loader: Iterable, teachers: Sequence[Model], device: torch.device) -> None:
        self.loader = loader
        self.teachers = teachers
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for batch in self.loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X_cpu, _label = batch
            else:
                X_cpu, _Xctl, _label = batch

            X_cpu = X_cpu.to(dtype=torch.float32)
            X_device = X_cpu.to(self.device, non_blocking=True)

            with torch.no_grad():
                teacher_profile_counts, _teacher_log_counts = _teacher_batch(
                    self.teachers, X_device, self.device
                )

            # Round to integer counts like file 2 does
            teacher_profile_counts = torch.round(teacher_profile_counts)
            
            # Ensure non-zero total counts (fit_generator requires this for MNLL)
            totals = teacher_profile_counts.reshape(teacher_profile_counts.shape[0], -1).sum(dim=1)
            zero_mask = totals == 0
            if zero_mask.any():
                # Add a small count to first position for zero-sum examples
                teacher_profile_counts[zero_mask, 0, 0] = 1.0

            teacher_profile_counts = teacher_profile_counts.cpu()
            mask = torch.ones_like(teacher_profile_counts, dtype=torch.bool)

            yield X_cpu, teacher_profile_counts, mask


def _build_validation_arrays(
    loader: Iterable,
    teachers: Sequence[Model],
    device: torch.device,
    n_batches: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build validation arrays from streaming loader.
    
    Returns (X_valid, y_valid) as numpy arrays.
    """
    X_list, y_list = [], []
    
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
            
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            X_cpu, _label = batch
        else:
            X_cpu, _Xctl, _label = batch

        X_cpu = X_cpu.to(dtype=torch.float32)
        X_device = X_cpu.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_profile_counts, _ = _teacher_batch(teachers, X_device, device)

        # Round to integer counts
        teacher_profile_counts = torch.round(teacher_profile_counts)
        
        # Ensure non-zero totals
        totals = teacher_profile_counts.reshape(teacher_profile_counts.shape[0], -1).sum(dim=1)
        zero_mask = totals == 0
        if zero_mask.any():
            teacher_profile_counts[zero_mask, 0, 0] = 1.0

        X_list.append(X_cpu.numpy())
        y_list.append(teacher_profile_counts.cpu().numpy())

    X_valid = np.concatenate(X_list, axis=0).astype(np.float32)
    y_valid = np.concatenate(y_list, axis=0).astype(np.float32)
    
    return X_valid, y_valid


def run_streaming_training(
    cell_type: str = "K562",
    timestamps: Sequence[str] = (),
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    mutation_rate: float = 0.04,
    sv_rate: float = 1.0,
    seed: int = 42,
    validation_iter: int = 100,
    early_stop_epochs: int = 10,
    n_val_batches: int = 10,
    verbose: bool = True,
    metrics_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
):
    """Train student model using streaming data from DistillerPeakGenerator.
    
    Parameters
    ----------
    cell_type : str
        Cell type for data loading (e.g., "K562")
    timestamps : Sequence[str]
        Teacher model timestamps
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
    alpha : float
        Weight for count loss term (total_loss = mnll + alpha * mse)
    mutation_rate : float
        Point mutation rate for augmentation
    sv_rate : float
        Structural variation rate (Poisson lambda)
    seed : int
        Random seed
    validation_iter : int
        Validate every N iterations
    early_stop_epochs : int
        Stop if no improvement for this many epochs
    n_val_batches : int
        Number of batches to use for validation set
    verbose : bool
        Enable per-epoch logging
    metrics_path : Path, optional
        Path to write training metrics TSV
    out_dir : Path, optional
        Output directory for model checkpoints
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or (proj_root / "models" / "distilled_student_streaming")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Student model
    trimming = (2114 - 1000) // 2
    model_save_path = out_dir / "student.model"
    
    student = Model(
        model_save_path=str(model_save_path),
        n_filters=512,
        n_layers=8,
        n_outputs=2,
        alpha=alpha,
        trimming=trimming,
    ).to(device)

    # Teachers (kept on CPU, moved to GPU one at a time during inference)
    if not timestamps:
        timestamps = tuple(DEFAULT_TIMESTAMPS)
    teachers = _load_teacher_models(timestamps, cell_type)

    # Training data loader
    train_loader = _make_streaming_loader(
        cell_type=cell_type,
        batch_size=batch_size,
        negative_ratio=0.125,
        max_jitter=1024,
        reverse_complement=True,
        mutation_rate=mutation_rate,
        sv_rate=sv_rate,
        seed=seed,
        verbose=verbose,
    )

    # Validation data loader (no augmentation for consistency)
    val_loader = _make_streaming_loader(
        cell_type=cell_type,
        batch_size=batch_size,
        negative_ratio=0.125,
        max_jitter=0,
        reverse_complement=False,
        mutation_rate=0.0,
        sv_rate=0.0,
        seed=seed + 1,
        verbose=False,
    )

    # Build validation arrays
    print(f"Building validation set ({n_val_batches} batches)...")
    X_valid, y_valid = _build_validation_arrays(val_loader, teachers, device, n_val_batches)
    print(f"Validation set: {X_valid.shape[0]} examples")

    # Wrap training loader with teacher distillation
    training_data = TeacherDistillationGenerator(train_loader, teachers, device)

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    ensure_parent_dir_exists(str(model_save_path))
    
    if verbose:
        arch_txt = model_save_path.with_suffix(".arch.txt")
        student.save_model_arch_to_txt(str(arch_txt))

    # Train using fit_generator (handles MNLL + MSE loss logging internally)
    student.fit_generator(
        training_data=training_data,
        optimizer=optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        max_epochs=epochs,
        batch_size=batch_size,
        validation_iter=validation_iter,
        early_stop_epochs=early_stop_epochs,
        verbose=verbose,
        save=True,
    )

    # Save metrics to custom path if specified
    if metrics_path and student.train_metrics:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as handle:
            for line in student.train_metrics:
                handle.write(line + "\n")
        print(f"Saved metrics to {metrics_path}")

    # Save additional checkpoint formats
    torch.save({"epoch": epochs, "model": student.state_dict()}, out_dir / "student_final.pt")
    print(f"Saved student to {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K562 streaming distillation training")
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--timestamps", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for count loss term")
    parser.add_argument("--validation-iter", type=int, default=100, help="Validate every N iterations")
    parser.add_argument("--early-stop-epochs", type=int, default=10)
    parser.add_argument("--n-val-batches", type=int, default=10, help="Number of batches for validation set")
    parser.add_argument("--mutation-rate", type=float, default=0.04, help="Point mutation rate for augmentation")
    parser.add_argument("--sv-rate", type=float, default=1.0, help="Structural variation rate (Poisson lambda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None, help="Path to write training metrics (tsv)")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose training logs")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    run_streaming_training(
        cell_type=args.cell_type,
        timestamps=tuple(args.timestamps) if args.timestamps else tuple(DEFAULT_TIMESTAMPS),
        epochs=max(1, args.epochs),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        mutation_rate=args.mutation_rate,
        sv_rate=args.sv_rate,
        seed=args.seed,
        validation_iter=args.validation_iter,
        early_stop_epochs=args.early_stop_epochs,
        n_val_batches=args.n_val_batches,
        verbose=not args.quiet,
        metrics_path=args.metrics_path,
        out_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()