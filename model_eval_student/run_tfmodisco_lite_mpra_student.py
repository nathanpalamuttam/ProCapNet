#!/usr/bin/env python3
"""
Run TF-MoDISco Lite on a student-only MPRA attribution file.

This script is intentionally strict:
- it only accepts a student per-model attribution NPZ, not an aggregate file
- it requires a completed MPRA metrics JSON before motif discovery starts
- it validates MPRA FASTA order/count/length against the attribution tensor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import modiscolite
import modiscolite.tfmodisco

from mpra_utils import (
    ensure_writable_paths,
    load_mpra_one_hot,
    load_npz_array,
    load_npz_meta,
    validate_paired_fasta_headers,
)


def summarize_modisco(h5_path: Path) -> list[dict]:
    rows = []
    with h5py.File(h5_path, "r") as handle:
        for sign in ("pos_patterns", "neg_patterns"):
            if sign not in handle:
                continue
            for pattern_name in handle[sign].keys():
                pattern = handle[sign][pattern_name]
                rows.append(
                    {
                        "metacluster": sign,
                        "pattern": pattern_name,
                        "n_seqlets": int(len(pattern["seqlets"]["sequence"])),
                        "cwm_len": int(pattern["contrib_scores"].shape[0]),
                    }
                )
    rows.sort(key=lambda row: (row["metacluster"], -row["n_seqlets"]))
    return rows


def validate_student_attribution_file(
    attributions_path: Path,
    *,
    student_model_stem: str,
) -> tuple[np.ndarray, dict]:
    attrs = load_npz_array(attributions_path)
    meta = load_npz_meta(attributions_path)

    if attrs.ndim != 3 or attrs.shape[1] != 4:
        raise ValueError(
            f"Expected attribution tensor shape (N, 4, L), found {attrs.shape}"
        )

    if meta.get("input_mode") not in (None, "mpra_fasta"):
        raise ValueError(
            f"TF-MoDISco MPRA script only accepts mpra_fasta attributions; "
            f"found input_mode={meta.get('input_mode')!r}"
        )

    models = meta.get("models", [])
    aggregate_models = meta.get("aggregate_models", [])
    if aggregate_models and len(aggregate_models) != 1:
        raise ValueError(
            "Refusing to run TF-MoDISco on an aggregate attribution file; "
            f"aggregate_models={aggregate_models}"
        )
    if len(models) > 1:
        raise ValueError(
            "Refusing to run TF-MoDISco on a multi-model attribution file; "
            f"models={models}"
        )

    filename_stem = attributions_path.stem
    student_from_name = filename_stem.endswith(f"__{student_model_stem}")
    student_from_meta = (
        len(models) == 1 and Path(models[0]).stem == student_model_stem
    )
    if not (student_from_name or student_from_meta):
        raise ValueError(
            f"Attribution file must be the student per-model NPZ; got {attributions_path}"
        )

    return attrs, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributions", required=True, type=Path)
    parser.add_argument("--mpra-fasta", required=True, type=Path)
    parser.add_argument("--paired-mpra-fasta", default=None, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-length", type=int, default=2114)
    parser.add_argument("--student-model-stem", default="student")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_h5 = args.out_dir / "tfmodisco_lite_results.h5"
    out_tsv = args.out_dir / "tfmodisco_pattern_summary.tsv"
    ensure_writable_paths([out_h5, out_tsv], overwrite=bool(args.overwrite))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    attrs, attr_meta = validate_student_attribution_file(
        args.attributions,
        student_model_stem=args.student_model_stem,
    )

    with open(args.metrics_json) as handle:
        metrics = json.load(handle)
    if metrics.get("status") != "success":
        raise ValueError(
            f"Metrics JSON must indicate successful completion before TF-MoDISco: {args.metrics_json}"
        )
    if int(metrics.get("input_record_count", -1)) != int(attrs.shape[0]):
        raise ValueError(
            "Metrics JSON input_record_count does not match attribution entries: "
            f"{metrics.get('input_record_count')} vs {attrs.shape[0]}"
        )

    headers, one_hot = load_mpra_one_hot(
        args.mpra_fasta,
        expected_length=args.expected_length,
    )
    if len(headers) != int(attrs.shape[0]):
        raise ValueError(
            f"FASTA record count {len(headers)} does not match attribution entries {attrs.shape[0]}"
        )

    if args.paired_mpra_fasta is not None:
        paired_headers, paired_one_hot = load_mpra_one_hot(
            args.paired_mpra_fasta,
            expected_length=args.expected_length,
        )
        validate_paired_fasta_headers(headers, paired_headers, label_a="mpra", label_b="paired_mpra")
        if paired_one_hot.shape[0] != one_hot.shape[0]:
            raise ValueError(
                f"mpra/paired_mpra record counts differ: {one_hot.shape[0]} vs {paired_one_hot.shape[0]}"
            )

    if one_hot.shape[2] != int(args.expected_length):
        raise ValueError(
            f"FASTA one-hot length {one_hot.shape[2]} does not match expected length {args.expected_length}"
        )

    hypothetical_contribs = np.swapaxes(attrs, 1, 2)
    one_hot_len_last = np.swapaxes(one_hot, 1, 2)
    if hypothetical_contribs.shape != one_hot_len_last.shape:
        raise ValueError(
            "One-hot sequence tensor and attribution tensor do not align: "
            f"{one_hot_len_last.shape} vs {hypothetical_contribs.shape}"
        )

    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=hypothetical_contribs,
        one_hot=one_hot_len_last,
        max_seqlets_per_metacluster=1000000,
        sliding_window_size=20,
        flank_size=5,
        target_seqlet_fdr=0.05,
        n_leiden_runs=50,
    )
    modiscolite.io.save_hdf5(str(out_h5), pos_patterns, neg_patterns)

    summary_rows = summarize_modisco(out_h5)
    with open(out_tsv, "w") as handle:
        handle.write("metacluster\tpattern\tn_seqlets\tcwm_len\n")
        for row in summary_rows:
            handle.write(
                f"{row['metacluster']}\t{row['pattern']}\t{row['n_seqlets']}\t{row['cwm_len']}\n"
            )

    print(f"Student-only TF-MoDISco results: {out_h5}", flush=True)
    print(f"Pattern summary: {out_tsv}", flush=True)
    if attr_meta:
        print(f"Attribution metadata source: {args.attributions}", flush=True)


if __name__ == "__main__":
    main()
