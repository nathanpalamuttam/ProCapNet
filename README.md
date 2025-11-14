# ProCapNet Model Distillation

This directory contains the implementation of knowledge distillation for ProCapNet models, enabling the compression of an ensemble of teacher models into a single, efficient student model.

## Overview

Knowledge distillation is a model compression technique that transfers knowledge from a larger, more complex model (or ensemble of models) to a smaller, more efficient model. In the context of ProCapNet:

- **Teacher Models**: An ensemble of 7 trained ProCapNet models that predict transcription initiation profiles
- **Student Model**: A single BPNet model that learns to replicate the ensemble's predictions
- **Key Advantage**: Achieves near-ensemble performance with significantly reduced computational cost and memory footprint

## Architecture

### Teacher Ensemble
- 7 independently trained ProCapNet models (default timestamps from 2023-05-29 to 2023-05-30)
- Location: `models/procap/{cell_type}/strand_merged_umap/{timestamp}.model`
- Each teacher outputs:
  - Profile logits: Shape `(batch_size, n_strands, out_window)`
  - Log counts: Shape `(batch_size, 1)`

### Student Model
- Based on BPNet architecture (from `BPNet_strand_merged_umap.py`)
- Default configuration:
  - Filters: 512
  - Layers: 8
  - Input window: 2114 bp
  - Output window: 1000 bp
  - Trimming: 557 bp per side

## Training Methodology

Two training approaches are available:

### 1. Streaming Training (`distill_k562.py`)
Real-time distillation where the student learns directly from teacher predictions generated on-the-fly.

**Advantages:**
- Memory efficient - no need to pre-generate and store predictions
- Supports data augmentation (point mutations, structural variations)
- Dynamic negative sampling with background suppression

**Key Features:**
- Point mutations: Randomly mutates bases at specified rate (default: 4%)
- Structural variations: Random insertions/deletions using Poisson rate (default: λ=1.0)
- Jitter augmentation: Random shifts up to 1024 bp
- Reverse complement augmentation
- Negative ratio: 0.125 (1 negative for every 8 positives)

### 2. Batch Training (`train_distilled_student.py`)
Trains on pre-computed teacher predictions stored in NPZ archives.

**Advantages:**
- Faster training iterations (no teacher inference overhead)
- Reproducible training data
- Easier validation and analysis

**Data Format:**
NPZ archive containing:
- `inputs`: DNA sequences (one-hot encoded)
- `teacher_log_probs`: Normalized log probabilities from ensemble
- `teacher_log_counts`: Log total counts from ensemble
- `teacher_profile_counts`: Profile counts (actual tracks)

## Loss Functions

The distillation loss combines two components:

### 1. Profile Loss (MNLL)
```python
MNLLLoss(student_log_probs, teacher_profile_counts)
```

**MNLL**: Multinomial Negative Log-Likelihood  
Treats teacher profile counts as pseudo-ground-truth and measures how well student predictions match the teacher distribution across positions.

### 2. Count Loss (log1pMSE)
```python
log1pMSELoss(student_log_counts, teacher_total_counts)
```

Mean squared error on `log(1 + x)` transformed counts.  
Ensures accurate total count predictions.  
**Default weight:** 0.1  

**Total Loss**
```python
total_loss = profile_loss + count_loss_weight * count_loss
```

---

## Usage

### Streaming Training
```bash
python src/npp8_files/distill_k562.py \
  --cell-type K562 \
  --epochs 100 \
  --batch-size 64 \
  --count-loss-weight 0.1 \
  --bg-suppress-weight 0.1 \
  --mutation-rate 0.04 \
  --sv-rate 1.0 \
  --seed 42 \
  --output-dir models/distilled_student_streaming
```

**Key Arguments:**
- `--cell-type`: Cell type to train on (default: K562)
- `--timestamps`: Teacher model timestamps (uses defaults if not specified)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 64)
- `--count-loss-weight`: Weight for count loss component (default: 0.1)
- `--bg-suppress-weight`: Weight for background suppression (default: 0.1)
- `--mutation-rate`: Point mutation rate for augmentation (default: 0.04)
- `--sv-rate`: Structural variation rate for augmentation (default: 1.0)

### Batch Training
```bash
python src/npp8_files/train_distilled_student.py \
  --archive data/procap/processed/K562/distillation/distillation_dataset_k562.npz \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --count-loss-weight 0.1 \
  --n-filters 512 \
  --n-layers 8 \
  --output-dir models/distilled_student
```

**Key Arguments:**
- `--archive`: Path to distillation NPZ archive
- `--val-fraction`: Fraction of data for validation (default: 0.1)
- `--train-limit`: Optional cap on training samples
- `--val-limit`: Optional cap on validation samples
- `--n-filters`: Number of convolutional filters (default: 512)
- `--n-layers`: Number of convolutional layers (default: 8)
- `--experiment-npz`: Optional NPZ with experimental profiles for validation
- `--eval-base-only`: Restrict validation to non-augmented examples

---

## Data Requirements

**Input Data**
- Genome: `genomes/hg38.withrDNA.fasta`
- Peaks: `data/procap/processed/{cell_type}/peaks.bed.gz`
- Negatives: `data/procap/processed/{cell_type}/dnase_peaks_no_procap_overlap.bed.gz`
- Teacher Models: Pre-trained models in `models/procap/{cell_type}/strand_merged_umap/`

**Data Preparation**
The `DistillerPeakGenerator` handles:
- Peak sampling with jitter augmentation
- Negative sampling from DNase peaks
- One-hot encoding of sequences
- Reverse complement augmentation
- Dynamic batch generation

---

## Output Files

Training produces:
- **Student Model:** `student_best.pt` (best validation checkpoint)
- **Final Model:** `student_last.pt` (final epoch checkpoint)
- **Training Metrics:** `training_metrics.jsonl` (per-epoch metrics)

**Metrics logged per epoch:**
- `train_prob`: Training profile loss (MNLL)
- `train_count`: Training count loss (log1pMSE)
- `val_prob`: Validation profile loss
- `val_count`: Validation count loss
- `val_total`: Combined validation loss

Optional experimental metrics (if `--experiment-npz` provided):
- `exp_profile_pearson_mean/median`: Profile Pearson correlation
- `exp_count_pearson_mean/median`: Count Pearson correlation
- `exp_count_log1pMSE_mean/median`: Count prediction error

---

## Key Implementation Details

### Ensemble Averaging
Teachers are averaged at the profile count level (not probability level):

```python
# For each teacher:
profile_counts = probabilities * total_counts

# Average across ensemble:
avg_profile_counts = mean(teacher_profile_counts)
avg_total_counts = mean(teacher_total_counts)
```

This approach maintains consistency between profile and count predictions.

### Label-Guided Training
Optional background suppression using binary labels:

- `labels == 1`: PRO-cap peaks (compute full loss)
- `labels == 0`: DNase peaks with no PRO-cap signal (optional background suppression)

When background suppression is enabled, profile loss is computed only on positive examples.

---

## Optimization

- Optimizer: **Adam**
- Learning Rate: `1e-4` (default)
- Gradient Management: `set_to_none=True` for memory efficiency
- Device: Automatically uses CUDA if available

---

## Performance Considerations

**Memory Usage**
- Streaming: Lower memory footprint, slower per-epoch
- Batch: Higher memory (loads NPZ), faster per-epoch

**Training Speed**
- Streaming: ~10–15 minutes per epoch (GPU-dependent)
- Batch: ~5–8 minutes per epoch (GPU-dependent)

---

## Recommended Settings

**For best quality:**
- Streaming with augmentation
- 100+ epochs
- `mutation_rate=0.04`, `sv_rate=1.0`

**For fast prototyping:**
- Batch training
- 5–10 epochs
- Smaller validation set with `--val-limit`
