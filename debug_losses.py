"""
Diagnostic script to understand MNLL vs count_loss behavior.
Add this code to your training loop to debug the issue.
"""

import torch

def diagnose_losses(
    student_logits,
    student_log_counts,
    teacher_profile_counts,
    teacher_log_counts,
    labels,
    epoch,
    batch_idx
):
    """Print diagnostic information about losses and gradients."""

    # Flatten
    student_logits_flat = student_logits.reshape(student_logits.shape[0], -1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits_flat, dim=-1)
    teacher_counts_flat = teacher_profile_counts.reshape(teacher_profile_counts.shape[0], -1)

    # Filter to peaks
    peak_mask = labels == 1
    num_peaks = peak_mask.sum().item()
    num_bg = (~peak_mask).sum().item()

    print(f"\n=== Epoch {epoch}, Batch {batch_idx} ===")
    print(f"Batch composition: {num_peaks} peaks, {num_bg} background")

    if num_peaks > 0:
        # Check teacher statistics on peaks
        teacher_counts_peaks = teacher_counts_flat[peak_mask]
        teacher_total_peaks = torch.exp(teacher_log_counts[peak_mask]) - 1.0

        print(f"\nTeacher statistics (peaks only):")
        print(f"  Profile counts - min: {teacher_counts_peaks.min():.4f}, max: {teacher_counts_peaks.max():.4f}, mean: {teacher_counts_peaks.mean():.4f}")
        print(f"  Total counts - min: {teacher_total_peaks.min():.4f}, max: {teacher_total_peaks.max():.4f}, mean: {teacher_total_peaks.mean():.4f}")
        print(f"  Sum of profile counts per example: {teacher_counts_peaks.sum(dim=-1).mean():.4f}")

        # Check student statistics on peaks
        student_probs_peaks = torch.exp(student_log_probs[peak_mask])
        student_total_peaks = torch.exp(student_log_counts[peak_mask]) - 1.0
        student_profile_counts_peaks = student_probs_peaks * student_total_peaks

        print(f"\nStudent statistics (peaks only):")
        print(f"  Predicted profile counts - min: {student_profile_counts_peaks.min():.4f}, max: {student_profile_counts_peaks.max():.4f}, mean: {student_profile_counts_peaks.mean():.4f}")
        print(f"  Predicted total counts - min: {student_total_peaks.min():.4f}, max: {student_total_peaks.max():.4f}, mean: {student_total_peaks.mean():.4f}")
        print(f"  Sum of profile counts per example: {student_profile_counts_peaks.sum(dim=-1).mean():.4f}")

        # Check how well distributions match
        # KL divergence between teacher and student profile distributions
        teacher_probs_peaks = teacher_counts_peaks / (teacher_counts_peaks.sum(dim=-1, keepdim=True) + 1e-12)
        kl_div = (teacher_probs_peaks * (torch.log(teacher_probs_peaks + 1e-12) - student_log_probs[peak_mask])).sum(dim=-1).mean()
        print(f"\nKL divergence (teacher || student) on peaks: {kl_div:.4f}")

        # Count MSE
        count_mse = ((teacher_total_peaks - student_total_peaks) ** 2).mean()
        print(f"Count MSE on peaks: {count_mse:.4f}")

        # Check for numerical issues
        print(f"\nNumerical health:")
        print(f"  Any NaN in student_log_probs: {torch.isnan(student_log_probs).any()}")
        print(f"  Any Inf in student_log_probs: {torch.isinf(student_log_probs).any()}")
        print(f"  Any NaN in teacher_counts: {torch.isnan(teacher_counts_peaks).any()}")
        print(f"  Any negative teacher_counts: {(teacher_counts_peaks < 0).any()}")

        # Check lgamma numerical stability
        lgamma_teacher = torch.lgamma(teacher_counts_peaks + 1)
        print(f"  Any NaN in lgamma(teacher_counts + 1): {torch.isnan(lgamma_teacher).any()}")
        print(f"  Any Inf in lgamma(teacher_counts + 1): {torch.isinf(lgamma_teacher).any()}")


# To use this, add to your training loop in distill_k562.py around line 329:
"""
if batch_idx == 0:  # First batch of each epoch
    from debug_losses import diagnose_losses
    diagnose_losses(
        student_logits, student_log_counts,
        teacher_profile_counts, teacher_log_counts,
        y, epoch, batch_idx
    )
"""
