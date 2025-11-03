#!/usr/bin/env python3
"""Compare teacher ensemble predictions to student predictions on the distillation archive."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-npz', type=Path, required=True,
                        help='Distillation NPZ containing teacher predictions (teacher_* arrays).')
    parser.add_argument('--student-npz', type=Path, required=True,
                        help='NPZ produced by predict_student_k562.py containing student_* arrays.')
    parser.add_argument('--output-tsv', type=Path, required=True,
                        help='Path to write per-example comparison metrics (TSV).')
    parser.add_argument('--experiment-npz', type=Path, default=None,
                        help='Optional NPZ containing experimental profile counts aligned with the distillation archive.')
    parser.add_argument('--summary-only', action='store_true',
                        help='Print aggregate statistics without writing the per-example TSV.')
    parser.add_argument(
        '--pearson-profile-plot',
        type=Path,
        default=None,
        help='Optional path to save per-position Pearson correlation curve (student vs experiment).',
    )
    return parser.parse_args()


def _flatten_profiles(arr: np.ndarray) -> np.ndarray:
    # arr shape: (n, strands, length)
    return arr.reshape(arr.shape[0], -1)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) and np.allclose(y, y[0]):
        return float('nan')
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(pearsonr(x, y)[0])


def log1p_mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((np.log1p(pred) - np.log1p(true)) ** 2))

def _pearson_curve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return Pearson correlation per position across the first axis."""
    if a.shape != b.shape:
        raise ValueError(f"Array shape mismatch for pearson curve: {a.shape} vs {b.shape}")

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)

    numerator = np.sum(a_centered * b_centered, axis=0)
    denom_a = np.sqrt(np.sum(a_centered ** 2, axis=0))
    denom_b = np.sqrt(np.sum(b_centered ** 2, axis=0))
    denom = denom_a * denom_b

    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.divide(
            numerator,
            denom,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=denom > 0,
        )
    return corr

def _compute_profile_curve(
    student_profiles: np.ndarray,
    experiment_profiles: np.ndarray,
) -> np.ndarray:
    """Compute per-position Pearson r across all examples for each strand."""
    n_strands = student_profiles.shape[1]
    curves = []
    for strand in range(n_strands):
        student = student_profiles[:, strand, :]
        experiment = experiment_profiles[:, strand, :]
        curves.append(_pearson_curve(student, experiment))
    return np.stack(curves, axis=0)

def main() -> None:
    args = parse_args()

    teacher = np.load(args.teacher_npz, mmap_mode='r')
    student = np.load(args.student_npz, mmap_mode='r')

    teacher_profiles = teacher['teacher_profile_counts']
    teacher_totals = teacher_profiles.sum(axis=(1, 2))

    student_profiles = student['student_profile_counts']
    student_totals = student_profiles.sum(axis=(1, 2))

    experiment_profiles = None
    experiment_totals = None
    if args.experiment_npz is not None:
        experiment = np.load(args.experiment_npz, mmap_mode='r')
        experiment_profiles = experiment['experimental_profile_counts']
        experiment_totals = experiment_profiles.sum(axis=(1, 2))
        if experiment_profiles.shape != teacher_profiles.shape:
            raise ValueError(
                f"Profile shape mismatch: experiment {experiment_profiles.shape} vs teacher {teacher_profiles.shape}")

    if teacher_profiles.shape != student_profiles.shape:
        raise ValueError(f"Profile shape mismatch: teacher {teacher_profiles.shape} vs student {student_profiles.shape}")

    teacher_log_probs = teacher.get('teacher_log_probs')
    student_log_probs = student.get('student_log_probs')

    n = teacher_profiles.shape[0]

    rows = []
    profile_corrs = []
    count_corrs = []
    count_mses = []
    exp_profile_corrs = []
    exp_count_corrs = []
    exp_count_mses = []
    teacher_exp_profile_corrs = []
    teacher_exp_count_corrs = []
    teacher_exp_count_mses = []

    teacher_flat = _flatten_profiles(teacher_profiles)
    student_flat = _flatten_profiles(student_profiles)
    experiment_flat = _flatten_profiles(experiment_profiles) if experiment_profiles is not None else None

    for i in range(n):
        prof_corr = _safe_pearson(student_flat[i], teacher_flat[i])
        cnt_corr = float('nan')
        try:
            cnt_corr = float(pearsonr(student_profiles[i].sum(axis=0), teacher_profiles[i].sum(axis=0))[0])
        except Exception:
            pass

        cnt_mse = log1p_mse(student_totals[i], teacher_totals[i])

        profile_corrs.append(prof_corr)
        count_corrs.append(cnt_corr)
        count_mses.append(cnt_mse)

        row = {
            'index': i,
            'profile_pearson': prof_corr,
            'count_pearson': cnt_corr,
            'count_log1pMSE': cnt_mse,
            'teacher_total': teacher_totals[i],
            'student_total': student_totals[i],
        }

        if teacher_log_probs is not None and student_log_probs is not None:
            row['kl_teacher_vs_student'] = float(np.sum(np.exp(teacher_log_probs[i]) * (teacher_log_probs[i] - student_log_probs[i])))

        if experiment_profiles is not None and experiment_flat is not None:
            exp_prof_corr = _safe_pearson(student_flat[i], experiment_flat[i])
            exp_cnt_corr = float('nan')
            try:
                exp_cnt_corr = float(pearsonr(student_profiles[i].sum(axis=0), experiment_profiles[i].sum(axis=0))[0])
            except Exception:
                pass
            exp_cnt_mse = log1p_mse(student_totals[i], experiment_totals[i])

            profile_corrs_exp_teacher = _safe_pearson(teacher_flat[i], experiment_flat[i])
            cnt_corr_exp_teacher = float('nan')
            try:
                cnt_corr_exp_teacher = float(pearsonr(teacher_profiles[i].sum(axis=0), experiment_profiles[i].sum(axis=0))[0])
            except Exception:
                pass
            cnt_mse_exp_teacher = log1p_mse(teacher_totals[i], experiment_totals[i])

            exp_profile_corrs.append(exp_prof_corr)
            exp_count_corrs.append(exp_cnt_corr)
            exp_count_mses.append(exp_cnt_mse)

            teacher_exp_profile_corrs.append(profile_corrs_exp_teacher)
            teacher_exp_count_corrs.append(cnt_corr_exp_teacher)
            teacher_exp_count_mses.append(cnt_mse_exp_teacher)

            row.update({
                'student_vs_experiment_profile_pearson': exp_prof_corr,
                'student_vs_experiment_count_pearson': exp_cnt_corr,
                'student_vs_experiment_count_log1pMSE': exp_cnt_mse,
                'teacher_vs_experiment_profile_pearson': profile_corrs_exp_teacher,
                'teacher_vs_experiment_count_pearson': cnt_corr_exp_teacher,
                'teacher_vs_experiment_count_log1pMSE': cnt_mse_exp_teacher,
                'experiment_total': experiment_totals[i],
            })

        rows.append(row)

    profile_corrs = np.array(profile_corrs, dtype=np.float64)
    count_corrs = np.array(count_corrs, dtype=np.float64)
    count_mses = np.array(count_mses, dtype=np.float64)
    if experiment_profiles is not None:
        exp_profile_corrs = np.array(exp_profile_corrs, dtype=np.float64)
        exp_count_corrs = np.array(exp_count_corrs, dtype=np.float64)
        exp_count_mses = np.array(exp_count_mses, dtype=np.float64)
        teacher_exp_profile_corrs = np.array(teacher_exp_profile_corrs, dtype=np.float64)
        teacher_exp_count_corrs = np.array(teacher_exp_count_corrs, dtype=np.float64)
        teacher_exp_count_mses = np.array(teacher_exp_count_mses, dtype=np.float64)
        if args.pearson_profile_plot is not None:
            pearson_curves = _compute_profile_curve(student_profiles, experiment_profiles)
        else:
            pearson_curves = None
    else:
        pearson_curves = None

    print('Summary (ignoring NaNs):')
    print(f'  Profile Pearson mean: {np.nanmean(profile_corrs):.4f}, median: {np.nanmedian(profile_corrs):.4f}')
    print(f'  Count Pearson   mean: {np.nanmean(count_corrs):.4f}, median: {np.nanmedian(count_corrs):.4f}')
    print(f'  Count log1pMSE  mean: {np.nanmean(count_mses):.4f}, median: {np.nanmedian(count_mses):.4f}')
    if experiment_profiles is not None:
        print('Student vs Experiment:')
        print(f'  Profile Pearson mean: {np.nanmean(exp_profile_corrs):.4f}, median: {np.nanmedian(exp_profile_corrs):.4f}')
        print(f'  Count Pearson   mean: {np.nanmean(exp_count_corrs):.4f}, median: {np.nanmedian(exp_count_corrs):.4f}')
        print(f'  Count log1pMSE  mean: {np.nanmean(exp_count_mses):.4f}, median: {np.nanmedian(exp_count_mses):.4f}')
        print('Teacher vs Experiment:')
        print(f'  Profile Pearson mean: {np.nanmean(teacher_exp_profile_corrs):.4f}, median: {np.nanmedian(teacher_exp_profile_corrs):.4f}')
        print(f'  Count Pearson   mean: {np.nanmean(teacher_exp_count_corrs):.4f}, median: {np.nanmedian(teacher_exp_count_corrs):.4f}')
        print(f'  Count log1pMSE  mean: {np.nanmean(teacher_exp_count_mses):.4f}, median: {np.nanmedian(teacher_exp_count_mses):.4f}')

    if args.pearson_profile_plot is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to generate the Pearson profile plot. "
                "Install it or omit --pearson-profile-plot."
            ) from exc

    if pearson_curves is not None and args.pearson_profile_plot is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to generate the Pearson profile plot. "
                "Install it or omit --pearson-profile-plot."
            ) from exc

        output_path = args.pearson_profile_plot.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        positions = np.arange(pearson_curves.shape[-1])
        plt.figure(figsize=(10, 4))
        for strand_idx, strand_curve in enumerate(pearson_curves):
            plt.plot(positions, strand_curve, label=f"Strand {strand_idx}")
        plt.xlabel("Profile position (bp)")
        plt.ylabel("Pearson r (student vs experiment)")
        plt.ylim(-1.0, 1.0)
        plt.title("Per-position Pearson correlation")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved Pearson profile curve to {output_path}")

    elif args.pearson_profile_plot is not None:
        import csv

        input_path = args.pearson_profile_plot
        curves_raw: list[list[float]] = []
        with input_path.open('r', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            for row in reader:
                if 'student_vs_experiment_profile_pearson' not in row:
                    raise ValueError(
                        "Input TSV missing 'student_vs_experiment_profile_pearson' column required for plot."
                    )
                curves_raw.append(float(row['student_vs_experiment_profile_pearson']))

        raise NotImplementedError(
            "Reading the curve directly from TSV is not yet implemented. "
            "Re-run with --experiment-npz to generate the plot."
        )

    if args.summary_only:
        return

    import csv
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_tsv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='	')
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote per-example metrics to {args.output_tsv}')


if __name__ == '__main__':
    main()
