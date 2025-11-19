# Debugging MNLL Loss Increase Issue

## Problem
MNLL (profile) loss is increasing while train_count loss is decreasing during distillation training.

## Fix Applied (Commit ffe2d48)

### The Bug
In `src/npp8_files/distill_k562.py`, there was an **asymmetric loss computation**:
- MNLL loss was computed only on peaks (`labels == 1`)
- Count loss was computed on ALL samples (peaks + background)

### Why This Caused Issues
1. Count loss optimized over entire batch including easy background samples
2. Model learned to predict low counts everywhere to minimize count loss on backgrounds
3. This conflicted with MNLL which needed high, well-distributed counts on peaks
4. Result: count loss decreased (from easy background wins), MNLL increased (degraded peak predictions)

### The Fix
Both losses now use consistent filtering:
```python
if labels is not None:
    profile_loss = MNLLLoss(student_log_probs[labels == 1], teacher_counts_flat[labels == 1]).mean()
    count_loss = log1pMSELoss(student_log_counts[labels == 1], torch.exp(teacher_log_counts[labels == 1]) - 1.0).mean()
```

## If MNLL Still Increases After the Fix

### Possible Causes

1. **Loss Scale Imbalance**
   - MNLL and count_loss might have very different magnitude ranges
   - Count loss weight (0.1) might still be too high if count_loss has larger values
   - **Solution**: Check actual loss values and adjust `count_loss_weight`

2. **Gradient Magnitude Issues**
   - Even with proper weighting, gradient norms might differ significantly
   - Count loss gradients might dominate in shared encoder layers
   - **Solution**: Add gradient clipping or use adaptive weighting (see `suggested_fix_v2.py`)

3. **Learning Rate Too High**
   - Current LR: 1e-4
   - Model might be overshooting optimal points
   - **Solution**: Try reducing to 5e-5 or 1e-5

4. **Numerical Instability**
   - MNLL uses `lgamma` with floating-point teacher counts (not true integers)
   - Could cause numerical issues with very small count values
   - **Solution**: Add small epsilon to teacher counts or use alternative loss

5. **Optimization Dynamics**
   - Model might need more epochs for both losses to converge together
   - Initial phase might show this behavior before stabilizing
   - **Solution**: Train longer and monitor when/if MNLL starts decreasing

## Diagnostic Steps

### 1. Check Loss Values
Add diagnostic logging (see `debug_losses.py`):
```python
# In training loop after line 329 in distill_k562.py
if batch_idx == 0:  # First batch of each epoch
    from debug_losses import diagnose_losses
    diagnose_losses(
        student_logits, student_log_counts,
        teacher_profile_counts, teacher_log_counts,
        y, epoch, batch_idx
    )
```

### 2. Monitor These Metrics
- Actual loss values (not just trends)
- Gradient norms for both losses
- Number of peaks vs background per batch
- Teacher vs student count distributions

### 3. Try These Experiments

**Experiment A: Adjust Loss Weight**
```python
count_loss_weight=0.01  # Reduce from 0.1
# or
count_loss_weight=0.5   # Increase if count loss is too small
```

**Experiment B: Reduce Learning Rate**
```python
learning_rate=5e-5  # Down from 1e-4
```

**Experiment C: Add Gradient Clipping**
```python
# After line 338 in distill_k562.py
kd_loss.backward()
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)  # Add this
opt.step()
```

**Experiment D: Use Adaptive Weighting**
Replace `_distillation_loss` with `_distillation_loss_v3_adaptive` from `suggested_fix_v2.py`

## Expected Behavior After Fix

With the filtering fix applied:
- Both losses should decrease together (though not necessarily monotonically)
- Total loss should decrease
- If one loss increases temporarily, the other should compensate
- Both should stabilize after sufficient training

## Key Questions to Answer

1. **Have you retrained after the fix?** The bug fix only helps if you retrain from scratch
2. **What are the actual loss magnitudes?** E.g., MNLL=1000, count=0.1 vs MNLL=2, count=5
3. **When does MNLL start increasing?**
   - From epoch 1? (might be learning dynamics)
   - After initially decreasing? (might be overfitting or oscillation)
4. **What fraction of each batch is peaks?** Should be ~87.5% with negative_ratio=0.125

## Files Created for Debugging
- `debug_losses.py`: Diagnostic code to understand loss behavior
- `suggested_fix_v2.py`: Alternative loss formulations with better balancing
- `DEBUGGING_MNLL.md`: This file

## Next Steps

1. Retrain model with the fix applied (commit ffe2d48)
2. Add diagnostic logging from `debug_losses.py`
3. Monitor actual loss values, not just trends
4. If still problematic, try experiments A-D above
5. Report back with:
   - Actual loss values over epochs
   - Output from diagnostic logging
   - Training curves/plots if available
