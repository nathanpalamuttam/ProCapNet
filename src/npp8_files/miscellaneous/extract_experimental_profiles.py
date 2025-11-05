#!/usr/bin/env python3
"""
Extract experimental PRO-cap counts matching the distillation NPZ ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "procap" / "processed" / "K562"
DEFAULT_REFERENCE_FASTA = PROJECT_ROOT / "genomes" / "hg38.fasta"

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distillation-npz",
        type=Path,
        required=True,
        help="Existing distillation archive (contains inputs/origin_index/augmentation_id).",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        required=True,
        help="Destination NPZ for experimental profile counts.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        default=DEFAULT_REFERENCE_FASTA,
        help=(
            "Reference genome FASTA used to provide sequence context for region bounds. "
            "Defaults to genomes/hg38.fasta."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use if torch ops are needed (not required, but kept for symmetry).",
    )
    return parser.parse_args(argv)

def load_raw_bigwigs() -> tuple[Path, Path]:
    pos_path = RAW_ROOT / "merged.pos.bigWig"
    neg_path = RAW_ROOT / "merged.neg.bigWig"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError("Missing merged bigWigs (merged.pos/neg.bigWig).")
    return pos_path, neg_path

def resolve_reference_fasta(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        candidates = [(Path.cwd() / resolved).resolve(), (PROJECT_ROOT / resolved).resolve()]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Reference FASTA not found. Checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {resolved}")
    return resolved

def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    # Load distillation data with proper context management
    with np.load(args.distillation_npz, mmap_mode="r") as distillation:
        inputs = np.array(distillation["inputs"])
        origin_index = distillation.get("origin_index")
        if origin_index is not None:
            origin_index = np.array(origin_index)
        augmentation_id = distillation.get("augmentation_id")
        if augmentation_id is not None:
            augmentation_id = np.array(augmentation_id)

    num_examples = inputs.shape[0]
    print(f"Distillation inputs: {num_examples}, shape per example = {inputs.shape[1:]}")

    import sys
    sys.path.append(str(PROJECT_ROOT / "src" / "2_train_models"))
    sys.path.append(str(PROJECT_ROOT / "src" / "utils"))
    from data_loading import extract_loci  # type: ignore

    merged_pos, merged_neg = load_raw_bigwigs()
    reference_fasta = resolve_reference_fasta(args.reference_fasta)

    # You need the loci (Peaks) corresponding to origin_index; typically stored in the metadata
    # For example, as part of the distillation pipeline; adjust this path to your loci file
    loci_path = PROJECT_ROOT / "data" / "procap" / "processed" / "K562" / "peaks.bed.gz"
    if not loci_path.exists():
        raise FileNotFoundError(f"Please point to a BED file containing loci (eg peaks): {loci_path}")

    print("Extracting experimental counts ...")
    peaks = str(loci_path)  # adapt if `extract_loci` expects a list
    
    # Extract loci and handle the tuple return properly
    result = extract_loci(
        loci=peaks,
        sequences=str(reference_fasta),
        in_signals=[str(merged_pos), str(merged_neg)],
        chroms=None,
        in_window=inputs.shape[-1],
        out_window=1000,
        max_jitter=0,
        verbose=True,
    )

    if isinstance(result, torch.Tensor):
        raise ValueError(
            "Expected `extract_loci` to return in-signal tensors; double-check the inputs."
        )
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise ValueError(
            "extract_loci did not return input signal tensors as expected. "
            "Verify the loci and in_signals arguments."
        )

    # outputs ordered as [sequences, (optional signals), in_signals]; we only need profiles
    experimental_profiles = result[-1]
    # prevent keeping the large sequence tensor in memory longer than needed
    del result

    experimental_profiles = (
        experimental_profiles.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if experimental_profiles.shape[-1] != 1000:
        trim = (experimental_profiles.shape[-1] - 1000) // 2
        experimental_profiles = experimental_profiles[:, :, trim : trim + 1000]
    if origin_index is not None:
        if origin_index.shape[0] != num_examples:
            raise ValueError(
                "origin_index length does not match number of distillation examples: "
                f"{origin_index.shape[0]} vs {num_examples}"
            )
        max_idx = origin_index.max(initial=-1)
        if max_idx >= experimental_profiles.shape[0]:
            raise ValueError(
                "origin_index references loci outside the extracted set: "
                f"max index {max_idx}, available {experimental_profiles.shape[0]} profiles."
            )
        experimental_profiles = experimental_profiles[origin_index]

    if experimental_profiles.shape[0] != num_examples:
        raise ValueError(
            f"Profile count mismatch: got {experimental_profiles.shape[0]} rows (expected {num_examples}). "
            "Verify origin_index metadata or provide a loci file covering every example."
        )

    out_dict = {"experimental_profile_counts": experimental_profiles}
    if origin_index is not None:
        out_dict["origin_index"] = origin_index.astype(np.int64, copy=False)
    if augmentation_id is not None:
        out_dict["augmentation_id"] = augmentation_id.astype(np.int16, copy=False)

    output_path = args.output_npz.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out_dict)
    print(f"Wrote experimental profiles to {output_path}")

if __name__ == "__main__":
    main()
