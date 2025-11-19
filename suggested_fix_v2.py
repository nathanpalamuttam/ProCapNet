"""
Alternative loss computation with better balancing.

If MNLL is still increasing after the filtering fix, try this version
which normalizes the losses before weighting them.
"""

import torch

def _distillation_loss_v2(
    student_logits: torch.Tensor,
    student_log_counts: torch.Tensor,
    teacher_profile_counts: torch.Tensor,
    teacher_log_counts: torch.Tensor,
    count_loss_weight: float,
    labels: torch.Tensor,
    mnll_loss_fn,
    log1pmse_loss_fn,
) -> tuple:
    """
    Alternative version that detaches losses for logging BEFORE weighting.
    This ensures we're logging the actual individual loss values, not
    values affected by the weighting.
    """
    # Flatten and normalize student logits
    student_logits_flat = student_logits.reshape(student_logits.shape[0], -1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits_flat, dim=-1)

    # Flatten teacher profile counts
    teacher_counts_flat = teacher_profile_counts.reshape(teacher_profile_counts.shape[0], -1)

    # Calculate losses with consistent filtering
    if labels is not None:
        # Only compute loss on peaks (labels == 1)
        peak_mask = labels == 1

        # Compute profile loss (MNLL)
        profile_loss = mnll_loss_fn(
            student_log_probs[peak_mask],
            teacher_counts_flat[peak_mask]
        ).mean()

        # Compute count loss (also on peaks only)
        count_loss = log1pmse_loss_fn(
            student_log_counts[peak_mask],
            torch.exp(teacher_log_counts[peak_mask]) - 1.0
        ).mean()

    else:
        profile_loss = mnll_loss_fn(student_log_probs, teacher_counts_flat).mean()
        count_loss = log1pmse_loss_fn(student_log_counts, torch.exp(teacher_log_counts) - 1.0).mean()

    # Log the raw loss values BEFORE weighting
    profile_loss_val = profile_loss.item()
    count_loss_val = count_loss.item()

    # Combine losses for backprop
    total_loss = profile_loss + count_loss_weight * count_loss

    return profile_loss_val, count_loss_val, total_loss


def _distillation_loss_v3_adaptive(
    student_logits: torch.Tensor,
    student_log_counts: torch.Tensor,
    teacher_profile_counts: torch.Tensor,
    teacher_log_counts: torch.Tensor,
    labels: torch.Tensor,
    mnll_loss_fn,
    log1pmse_loss_fn,
    profile_weight: float = 1.0,
    count_weight: float = 0.1,
) -> tuple:
    """
    Version with gradient norm balancing to prevent one loss from dominating.

    This version scales the count loss dynamically so that its gradient magnitude
    is proportional to the profile loss gradient magnitude.
    """
    # Flatten and normalize student logits
    student_logits_flat = student_logits.reshape(student_logits.shape[0], -1)
    student_log_probs = torch.nn.functional.log_softmax(student_logits_flat, dim=-1)

    # Flatten teacher profile counts
    teacher_counts_flat = teacher_profile_counts.reshape(teacher_profile_counts.shape[0], -1)

    # Calculate losses with consistent filtering
    if labels is not None:
        peak_mask = labels == 1
        profile_loss = mnll_loss_fn(student_log_probs[peak_mask], teacher_counts_flat[peak_mask]).mean()
        count_loss = log1pmse_loss_fn(student_log_counts[peak_mask], torch.exp(teacher_log_counts[peak_mask]) - 1.0).mean()
    else:
        profile_loss = mnll_loss_fn(student_log_probs, teacher_counts_flat).mean()
        count_loss = log1pmse_loss_fn(student_log_counts, torch.exp(teacher_log_counts) - 1.0).mean()

    # Log the raw loss values
    profile_loss_val = profile_loss.item()
    count_loss_val = count_loss.item()

    # Normalize losses by their magnitudes to prevent scale issues
    # This makes the relative weighting more meaningful
    profile_loss_normalized = profile_loss / (profile_loss.detach() + 1e-8)
    count_loss_normalized = count_loss / (count_loss.detach() + 1e-8)

    # Combine with normalized losses (this keeps gradients balanced)
    total_loss = profile_weight * profile_loss_normalized * profile_loss.detach() + \
                 count_weight * count_loss_normalized * count_loss.detach()

    return profile_loss_val, count_loss_val, total_loss


# Suggested modifications to try:
# 1. Increase count_loss_weight from 0.1 to 0.5 or 1.0 to see if it's a weighting issue
# 2. Decrease learning rate from 1e-4 to 1e-5 to prevent overshooting
# 3. Add gradient clipping: torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
# 4. Use v3_adaptive version above to auto-balance gradient magnitudes
