#!/usr/bin/env python3
"""
Run TF-MoDISco on existing attribution scores and visualize attribution logos
for highly expressed promoter-overlap peaks.

Defaults are set up for the K562 student/distillation artifacts in this
repository, but the script can be reused by overriding paths on the CLI.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import os
import zipfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta

import modiscolite
import modiscolite.tfmodisco


IN_WINDOW = 2114
OUT_WINDOW = 1000
SLICE_LEN = 1000
DNA_ALPHABET = np.array(list("ACGT"))
DNA_TO_INDEX = {base: i for i, base in enumerate(DNA_ALPHABET)}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_out_dir = Path(__file__).resolve().parent / "tfmodisco_promoter_analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attributions",
        default=Path(__file__).resolve().parent / "attributions__ensemble_mean.npz",
        type=Path,
    )
    parser.add_argument(
        "--peaks",
        default=repo_root / "data/procap/processed/K562/peaks.bed.gz",
        type=Path,
    )
    parser.add_argument(
        "--genome",
        default=repo_root / "genomes/hg38.withrDNA.fasta",
        type=Path,
    )
    parser.add_argument(
        "--ccres",
        default=repo_root / "annotations/K562/cCREs.bed.gz",
        type=Path,
    )
    parser.add_argument(
        "--experimental-profiles",
        default=repo_root / "data/procap/processed/K562/distillation/experimental_profiles_k562.npz",
        type=Path,
    )
    parser.add_argument(
        "--predicted-profiles",
        default=repo_root / "data/procap/processed/K562/distillation/student_predictions_k562.npz",
        type=Path,
    )
    parser.add_argument(
        "--gtf",
        default=repo_root / "annotations/gencode.v41.annotation.gtf.gz",
        type=Path,
    )
    parser.add_argument("--out-dir", default=default_out_dir, type=Path)
    parser.add_argument("--top-n-promoters", default=12, type=int)
    parser.add_argument("--max-seqlets", default=1000000, type=int)
    parser.add_argument("--max-loci", default=None, type=int)
    parser.add_argument("--skip-tfmodisco", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_array_from_npz(npz_path: Path, key: str = "arr_0") -> np.ndarray:
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(f"{key}.npy") as handle:
            return np.load(handle)


def load_peaks(peaks_path: Path) -> pd.DataFrame:
    peaks = pd.read_csv(peaks_path, sep="\t", header=None)
    peaks = peaks.iloc[:, :8].copy()
    peaks.columns = [
        "chrom",
        "start",
        "end",
        "strand",
        "confidence",
        "peak_type",
        "summit_plus",
        "summit_minus",
    ]
    peaks["start"] = peaks["start"].astype(int)
    peaks["end"] = peaks["end"].astype(int)
    peaks["center"] = peaks["start"] + (peaks["end"] - peaks["start"]) // 2
    peaks["locus_id"] = np.arange(len(peaks))
    return peaks


def fetch_one_hot_sequences(peaks: pd.DataFrame, genome_path: Path, in_window: int) -> np.ndarray:
    half = in_window // 2
    fasta = Fasta(str(genome_path), rebuild=False)
    seqs = np.zeros((len(peaks), in_window, 4), dtype=np.int8)
    for i, row in peaks.iterrows():
        start = row.center - half
        end = row.center + half
        seq = fasta[row.chrom][start:end].seq.upper()
        if len(seq) != in_window:
            raise ValueError(f"Expected {in_window} bp at row {i}, got {len(seq)}")
        for j, base in enumerate(seq):
            idx = DNA_TO_INDEX.get(base)
            if idx is not None:
                seqs[i, j, idx] = 1
    return seqs


def center_slice(arr: np.ndarray, target_len: int) -> np.ndarray:
    width = arr.shape[1]
    start = width // 2 - target_len // 2
    end = start + target_len
    return arr[:, start:end, :]


def run_tfmodisco(one_hot: np.ndarray, scores: np.ndarray, max_seqlets: int, out_h5: Path) -> None:
    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=scores,
        one_hot=one_hot,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=20,
        flank_size=5,
        target_seqlet_fdr=0.05,
        n_leiden_runs=50,
    )
    modiscolite.io.save_hdf5(str(out_h5), pos_patterns, neg_patterns)


def load_pls_ccres(ccres_path: Path) -> pd.DataFrame:
    ccres = pd.read_csv(ccres_path, sep="\t", header=None, comment="#")
    ccres = ccres[[0, 1, 2, 3, 9]].copy()
    ccres.columns = ["chrom", "start", "end", "ccre_id", "annotation"]
    ccres = ccres[ccres["annotation"].astype(str).str.contains("PLS", regex=False)].copy()
    ccres["start"] = ccres["start"].astype(int)
    ccres["end"] = ccres["end"].astype(int)
    return ccres.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def mark_promoter_overlaps(peaks: pd.DataFrame, promoters: pd.DataFrame) -> pd.DataFrame:
    peaks = peaks.copy()
    peaks["is_promoter"] = False
    peaks["ccre_ids"] = ""

    promoters_by_chrom = {
        chrom: grp[["start", "end", "ccre_id"]].to_records(index=False)
        for chrom, grp in promoters.groupby("chrom", sort=False)
    }

    for chrom, grp in peaks.groupby("chrom", sort=False):
        promoter_intervals = promoters_by_chrom.get(chrom)
        if promoter_intervals is None:
            continue
        starts = np.array([x[0] for x in promoter_intervals], dtype=np.int64)
        j = 0
        grp_idx = grp.index.to_numpy()
        grp_starts = grp["start"].to_numpy()
        grp_ends = grp["end"].to_numpy()
        for idx, peak_start, peak_end in zip(grp_idx, grp_starts, grp_ends):
            while j < len(promoter_intervals) and promoter_intervals[j][1] <= peak_start:
                j += 1
            k = j
            hit_ids = []
            while k < len(promoter_intervals) and starts[k] < peak_end:
                if promoter_intervals[k][1] > peak_start:
                    hit_ids.append(promoter_intervals[k][2])
                k += 1
            if hit_ids:
                peaks.at[idx, "is_promoter"] = True
                peaks.at[idx, "ccre_ids"] = ",".join(hit_ids)
    return peaks


def aggregate_augmented_profiles(npz_path: Path, profile_key: str) -> np.ndarray:
    data = np.load(npz_path)
    profiles = data[profile_key].astype(np.float32)
    origin_index = data["origin_index"].astype(np.int64)
    n_loci = origin_index.max() + 1
    summed = np.zeros((n_loci,) + profiles.shape[1:], dtype=np.float32)
    counts = np.zeros(n_loci, dtype=np.int32)
    for idx, profile in zip(origin_index, profiles):
        summed[idx] += profile
        counts[idx] += 1
    summed /= counts[:, None, None]
    return summed


def parse_gtf_attributes(attr_str: str) -> dict[str, str]:
    out = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if not part or " " not in part:
            continue
        key, value = part.split(" ", 1)
        out[key] = value.strip().strip('"')
    return out


def load_tss_by_chrom(gtf_path: Path) -> dict[str, list[tuple[int, str]]]:
    tss_by_chrom: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            chrom, _, _, start, end, _, strand, _, attrs = fields
            attr_map = parse_gtf_attributes(attrs)
            gene_name = attr_map.get("gene_name") or attr_map.get("gene_id") or "NA"
            tss = int(start) if strand == "+" else int(end)
            tss_by_chrom[chrom].append((tss, gene_name))
    for chrom in tss_by_chrom:
        tss_by_chrom[chrom].sort()
    return tss_by_chrom


def annotate_nearest_gene(peaks: pd.DataFrame, tss_by_chrom: dict[str, list[tuple[int, str]]]) -> pd.DataFrame:
    peaks = peaks.copy()
    genes = []
    dists = []
    for row in peaks.itertuples(index=False):
        chrom_tss = tss_by_chrom.get(row.chrom, [])
        if not chrom_tss:
            genes.append("NA")
            dists.append(np.nan)
            continue
        positions = [x[0] for x in chrom_tss]
        pos = bisect.bisect_left(positions, row.center)
        candidates = []
        if pos < len(chrom_tss):
            candidates.append(chrom_tss[pos])
        if pos > 0:
            candidates.append(chrom_tss[pos - 1])
        best_tss, best_gene = min(candidates, key=lambda x: abs(x[0] - row.center))
        genes.append(best_gene)
        dists.append(abs(best_tss - row.center))
    peaks["nearest_gene"] = genes
    peaks["nearest_tss_distance"] = dists
    return peaks


def observed_contrib_matrix(one_hot_seq: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    obs = one_hot_seq * attrs
    return pd.DataFrame(obs, columns=list("ACGT"))


def plot_promoter_examples(
    selected: pd.DataFrame,
    one_hot_slice: np.ndarray,
    attrs_slice: np.ndarray,
    experimental_profiles: np.ndarray,
    predicted_profiles: np.ndarray,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    x = np.arange(one_hot_slice.shape[1]) - one_hot_slice.shape[1] // 2
    for row in selected.itertuples(index=False):
        idx = int(row.locus_id)
        obs_logo = observed_contrib_matrix(one_hot_slice[idx], attrs_slice[idx])
        exp_prof = experimental_profiles[idx]
        pred_prof = predicted_profiles[idx]

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 2.4], hspace=0.15)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(x, exp_prof[0], color="#c0392b", lw=1.2, label="Experimental +")
        ax0.plot(x, -exp_prof[1], color="#2980b9", lw=1.2, label="Experimental -")
        ax0.plot(x, pred_prof[0], color="#e67e22", lw=1.0, alpha=0.8, label="Predicted +")
        ax0.plot(x, -pred_prof[1], color="#16a085", lw=1.0, alpha=0.8, label="Predicted -")
        ax0.axvline(0, color="black", ls="--", lw=0.8, alpha=0.6)
        ax0.set_ylabel("PRO-cap")
        ax0.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)

        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        pos_attr = attrs_slice[idx].clip(min=0).sum(axis=1)
        neg_attr = attrs_slice[idx].clip(max=0).sum(axis=1)
        ax1.fill_between(x, 0, pos_attr, color="#d35400", alpha=0.7)
        ax1.fill_between(x, 0, neg_attr, color="#1f78b4", alpha=0.7)
        ax1.axvline(0, color="black", ls="--", lw=0.8, alpha=0.6)
        ax1.set_ylabel("Attr sum")

        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        logomaker.Logo(obs_logo, ax=ax2, color_scheme="classic", baseline_width=0.0)
        ax2.axvline(one_hot_slice.shape[1] // 2, color="black", ls="--", lw=0.8, alpha=0.6)
        ax2.set_ylabel("Obs. contrib")
        ax2.set_xlabel("Position relative to locus center (bp)")

        title = (
            f"{row.nearest_gene} | {row.chrom}:{row.start}-{row.end} | "
            f"expr={row.expression_score:.1f} | dist_to_TSS={int(row.nearest_tss_distance)}"
        )
        fig.suptitle(title, y=0.98)
        fig.tight_layout()
        save_path = out_dir / (
            f"{idx:05d}_{row.nearest_gene}_{row.chrom}_{row.start}_{row.end}.png"
        )
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def summarize_modisco(h5_path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(h5_path, "r") as handle:
        for sign in ("pos_patterns", "neg_patterns"):
            if sign not in handle:
                continue
            grp = handle[sign]
            for pattern_name in grp.keys():
                pattern = grp[pattern_name]
                rows.append(
                    {
                        "metacluster": sign,
                        "pattern": pattern_name,
                        "n_seqlets": int(len(pattern["seqlets"]["sequence"])),
                        "cwm_len": int(pattern["contrib_scores"].shape[0]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["metacluster", "n_seqlets"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    print(f"Loading peaks from {args.peaks}", flush=True)
    peaks = load_peaks(args.peaks)
    if args.max_loci is not None:
        peaks = peaks.iloc[: args.max_loci].copy()

    print(f"Loading attributions from {args.attributions}", flush=True)
    attrs = load_array_from_npz(args.attributions)
    if args.max_loci is not None:
        attrs = attrs[: args.max_loci]
    if attrs.shape != (len(peaks), 4, IN_WINDOW):
        raise ValueError(f"Unexpected attribution shape {attrs.shape}; expected ({len(peaks)}, 4, {IN_WINDOW})")

    print(f"Extracting one-hot sequences from {args.genome}", flush=True)
    one_hot = fetch_one_hot_sequences(peaks, args.genome, IN_WINDOW)

    print("Preparing TF-MoDISco inputs", flush=True)
    one_hot_slice = center_slice(one_hot, SLICE_LEN)
    attrs_len_last = np.swapaxes(attrs, 1, 2)
    attrs_slice = center_slice(attrs_len_last, SLICE_LEN)

    modisco_h5 = args.out_dir / "tfmodisco_profile_like_results.h5"
    if args.skip_tfmodisco:
        print("Skipping TF-MoDISco run; generating promoter outputs only", flush=True)
    else:
        print(f"Running TF-MoDISco and saving to {modisco_h5}", flush=True)
        run_tfmodisco(one_hot_slice, attrs_slice, args.max_seqlets, modisco_h5)

    print("Loading promoter annotations from cCRE PLS intervals", flush=True)
    promoters = load_pls_ccres(args.ccres)
    peaks = mark_promoter_overlaps(peaks, promoters)

    print("Aggregating experimental and predicted profiles across augmentations", flush=True)
    experimental_profiles = aggregate_augmented_profiles(
        args.experimental_profiles, "experimental_profile_counts"
    )
    predicted_profiles = aggregate_augmented_profiles(
        args.predicted_profiles, "student_profile_counts"
    )
    if args.max_loci is not None:
        experimental_profiles = experimental_profiles[: args.max_loci]
        predicted_profiles = predicted_profiles[: args.max_loci]
    expression_score = experimental_profiles.sum(axis=(1, 2))
    peaks["expression_score"] = expression_score

    promoter_peaks = peaks[peaks["is_promoter"]].copy()
    promoter_peaks = promoter_peaks.sort_values("expression_score", ascending=False).reset_index(drop=True)

    print("Annotating nearest genes for top promoter examples", flush=True)
    top_promoters = promoter_peaks.head(args.top_n_promoters).copy()
    top_promoters = annotate_nearest_gene(top_promoters, load_tss_by_chrom(args.gtf))

    summary_tsv = args.out_dir / "top_promoters_by_expression.tsv"
    top_promoters.to_csv(summary_tsv, sep="\t", index=False)

    print(f"Writing promoter plots to {args.out_dir / 'promoter_plots'}", flush=True)
    plot_promoter_examples(
        top_promoters,
        one_hot_slice,
        attrs_slice,
        experimental_profiles,
        predicted_profiles,
        args.out_dir / "promoter_plots",
    )

    if modisco_h5.exists():
        print("Summarizing TF-MoDISco patterns", flush=True)
        modisco_summary = summarize_modisco(modisco_h5)
        modisco_summary.to_csv(args.out_dir / "tfmodisco_pattern_summary.tsv", sep="\t", index=False)

    print("Done.", flush=True)
    print(f"Top promoters table: {summary_tsv}", flush=True)
    if modisco_h5.exists():
        print(f"TF-MoDISco results: {modisco_h5}", flush=True)


if __name__ == "__main__":
    main()
