"""Utilities for K562 distillation.

Adds a streaming mode that trains a student directly from
DistillerPeakGenerator batches while matching a teacher ensemble and
optionally using binary labels (peak/background) to suppress background.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import torch

proj_root = Path(__file__).resolve().parents[2]
train_src = proj_root / "src" / "2_train_models"
utils_src = proj_root / "src" / "utils"
npp8_src = proj_root / "src" / "npp8_files"

import sys
for p in (train_src, utils_src, npp8_src):
    ps = str(p)
    if ps not in sys.path:
        sys.path.append(ps)

from BPNet_strand_merged_umap import Model  # type: ignore  # noqa: E402
from data_loader import DistillerPeakGenerator  # type: ignore  # noqa: E402
from misc import ensure_parent_dir_exists  # type: ignore  # noqa: E402
from losses import MNLLLoss, log1pMSELoss  # type: ignore  # noqa: E402



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

        model: Model = torch.load(model_path, map_location=device, weights_only = False)
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

        track_accum = None

        for model in models:
            with torch.no_grad():
                y_profile, y_counts = model(batch)
                flat = y_profile.reshape(y_profile.shape[0], -1)
                log_probs = log_softmax(flat).reshape_as(y_profile)
            
            log_probs_np = log_probs.detach().cpu().numpy()
            log_counts_np = y_counts.detach().cpu().numpy()
            
            probs_np = np.exp(log_probs_np).astype(np.float32, copy=False)
            total_counts_np = np.exp(log_counts_np).astype(np.float32, copy=False) - 1.0
            track_np = probs_np * total_counts_np.reshape(-1, 1, 1)

            if track_accum is None:
                track_accum = track_np
            else:
                track_accum += track_np

        # Average the tracks
        track_avg = track_accum / float(len(models))

        # Derive total_counts from averaged track
        total_counts_avg = track_avg.reshape(track_avg.shape[0], -1).sum(axis=1, keepdims=True) + 1.0

        # Derive probabilities from averaged track and total_counts
        prob_avg = track_avg / np.clip(total_counts_avg.reshape(-1, 1, 1) - 1.0, 1e-12, None)

        log_prob_out[start:end] = np.log(np.clip(prob_avg, 1e-12, None)).astype(np.float32, copy=False)
        log_count_out[start:end] = np.log(np.clip(total_counts_avg, 1e-12, None)).astype(np.float32, copy=False)
        profile_counts_out[start:end] = track_avg.astype(np.float32, copy=False)

    return log_prob_out, log_count_out, profile_counts_out


def _make_streaming_loader(
    cell_type: str,
    in_window: int = 2114,
    out_window: int = 1000,
    batch_size: int = 64,
    negative_ratio: float = 0.125,
    max_jitter: int = 1024,
    reverse_complement: bool = True,
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
    """Return (teacher_log_probs, teacher_log_counts) as torch tensors on device."""
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    prob_accum: Optional[torch.Tensor] = None
    total_accum: Optional[torch.Tensor] = None

    for m in models:
        m.eval()
        logits, log_counts = m(X)
        flat = logits.reshape(logits.shape[0], -1)
        log_probs = log_softmax(flat).reshape_as(logits)
        probs = torch.exp(log_probs)
        total = torch.exp(log_counts) - 1.0
        
        if prob_accum is None:
            prob_accum = probs
            total_accum = total
        else:
            prob_accum = prob_accum + probs
            total_accum = total_accum + total

    assert prob_accum is not None and total_accum is not None
    
    # Average the profiles and counts separately
    prob_avg = prob_accum / float(len(models))
    total_avg = total_accum / float(len(models))
    
    # Compute the track (profile counts) from averaged quantities
    profile_counts = prob_avg * total_avg.view(-1, 1, 1)
    
    teacher_log_counts = torch.log(torch.clamp(total_avg + 1.0, min=1e-12))
    return profile_counts, teacher_log_counts


def _distillation_loss(
    student_logits: torch.Tensor,
    student_log_counts: torch.Tensor,
    teacher_profile_counts: torch.Tensor,
    teacher_log_counts: torch.Tensor,
    count_loss_weight: float,
    labels: torch.Tensor
) -> Tuple[float, float, torch.Tensor]:
    """Compute distillation loss using mixture loss on teacher profile counts.
    
    Parameters
    ----------
    student_logits: torch.Tensor
        Student model logits (unnormalized)
    student_log_counts: torch.Tensor
        Student model log counts
    teacher_profile_counts: torch.Tensor
        Teacher ensemble profile counts (treated as "ground truth" y)
    teacher_log_counts: torch.Tensor
        Teacher ensemble log counts
    count_loss_weight: float
        Weight for count loss
    labels: torch.Tensor, optional
        Binary labels (1=peak, 0=background) for label-guided training
        
    Returns
    -------
    profile_loss: float
        Profile loss value (for logging)
    count_loss: float
        Count loss value (for logging)
    total_loss: torch.Tensor
        Combined loss for backpropagation
    """
    # Flatten and normalize student logits
    student_logits_flat = student_logits.reshape(student_logits.shape[0], -1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits_flat, dim=-1)
    
    # Flatten teacher profile counts (these are our "y" - the pseudo ground truth)
    teacher_counts_flat = teacher_profile_counts.reshape(teacher_profile_counts.shape[0], -1)
    
    # Calculate profile loss using MNLL
    if labels is not None:
        # Only compute loss on peaks (labels == 1)
        profile_loss = MNLLLoss(student_log_probs[labels == 1], teacher_counts_flat[labels == 1]).mean()
    else:
        profile_loss = MNLLLoss(student_log_probs, teacher_counts_flat).mean()
    
    # Calculate count loss
    count_loss = log1pMSELoss(student_log_counts, torch.exp(teacher_log_counts) - 1.0).mean()
    
    # Extract values for logging
    profile_loss_val = profile_loss.item()
    count_loss_val = count_loss.item()
    
    # Mix losses together
    total_loss = profile_loss + count_loss_weight * count_loss
    
    return profile_loss_val, count_loss_val, total_loss


def run_streaming_training(
    cell_type: str = "K562",
    timestamps: Sequence[str] = (),
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    count_loss_weight: float = 0.1,
    bg_suppress_weight: float = 0.1,
    mutation_rate: float = 0.04,
    sv_rate: float = 1.0,
    seed: int = 42,
    out_dir: Optional[Path] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Student model
    trimming = (2114 - 1000) // 2
    student = Model(
        model_save_path=str((out_dir or proj_root / "models" / "distilled_student_streaming") / "student_v2.model"),
        n_filters=512,
        n_layers=8,
        trimming=trimming,
    ).to(device)

    # Teachers
    if not timestamps:
        timestamps = tuple(DEFAULT_TIMESTAMPS)
    teachers = _load_teacher_models(timestamps, cell_type, device)

    # Data
    loader = _make_streaming_loader(
        cell_type=cell_type,
        batch_size=batch_size,
        negative_ratio=0.125,
        max_jitter=1024,
        reverse_complement=True,
        seed=seed,
        verbose=True,
    )

    opt = torch.optim.Adam(student.parameters(), lr=learning_rate)

    history: List[dict] = []
    for epoch in range(1, epochs + 1):
        student.train()
        epoch_prob, epoch_count, epoch_bg = [], [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X, y = batch
            else:
                X, _Xctl, y = batch  # controls path

            X = X.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device)

            # Get teacher predictions (profile counts, not log probs)
            with torch.no_grad():
                teacher_profile_counts, teacher_log_counts = _teacher_batch(teachers, X, device)

            # Get student predictions
            student_logits, student_log_counts = student(X)

            # Compute distillation loss using mixture loss
            profile_loss_val, count_loss_val, kd_loss = _distillation_loss(
                student_logits=student_logits,
                student_log_counts=student_log_counts,
                teacher_profile_counts=teacher_profile_counts,
                teacher_log_counts=teacher_log_counts,
                count_loss_weight=count_loss_weight,
                labels = y
            )
            opt.zero_grad(set_to_none=True)
            kd_loss.backward()
            opt.step()

            epoch_prob.append(profile_loss_val)
            epoch_count.append(count_loss_val)

        rec = {
            "epoch": epoch,
            "train_prob": float(np.mean(epoch_prob) if epoch_prob else 0.0),
            "train_count": float(np.mean(epoch_count) if epoch_count else 0.0),
        }
        if bg_suppress_weight > 0:
            rec["train_bg"] = float(torch.stack(epoch_bg).mean().item() if epoch_bg else 0.0)
            print(
                f"Epoch {epoch:02d}: MNLL(profile)={rec['train_prob']:.4f} "
                f"log1pMSE(count)={rec['train_count']:.4f} BG={rec['train_bg']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}: MNLL(profile)={rec['train_prob']:.4f} "
                f"log1pMSE(count)={rec['train_count']:.4f}"
            )
        history.append(rec)

    # Save student
    out_dir = out_dir or (proj_root / "models" / "distilled_student_streaming")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epochs, "model": student.state_dict()}, out_dir / "student_best.pt")
    with (out_dir / "training_metrics.jsonl").open("w") as handle:
        import json
        for row in history:
            handle.write(json.dumps(row) + "\n")
    print(f"Saved streaming student to {out_dir}")

def _stack_batches(batches: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(batches)
    if not arrays:
        raise ValueError("No arrays to stack")
    return np.concatenate(arrays, axis=0)


def _write_metadata(metadata_path: Path, metadata: dict) -> None:
    ensure_parent_dir_exists(str(metadata_path))
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

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
    parser.add_argument("--stream", action="store_true", help="Run streaming KD training instead of building NPZ")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--count-loss-weight", type=float, default=0.1)
    parser.add_argument("--bg-suppress-weight", type=float, default=0.1)
    parser.add_argument("--mutation-rate", type=float, default=0.04, help="Point mutation rate for augmentation")
    parser.add_argument("--sv-rate", type=float, default=1.0, help="Structural variation rate (Poisson lambda)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_streaming_training(
        cell_type=args.cell_type,
        timestamps=args.timestamps or tuple(DEFAULT_TIMESTAMPS),
        epochs=max(1, args.epochs),
        batch_size=args.batch_size,
        learning_rate=1e-4,
        count_loss_weight=args.count_loss_weight,
        bg_suppress_weight=args.bg_suppress_weight,
        mutation_rate=args.mutation_rate,
        sv_rate=args.sv_rate,
        seed=args.seed,
        out_dir=out_dir,
    )
    


if __name__ == "__main__":
    main()
