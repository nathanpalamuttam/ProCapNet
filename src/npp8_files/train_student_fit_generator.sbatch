#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:tP100:1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --account=danko_0001
#SBATCH --job-name=procap_student
#SBATCH --chdir=/home2/npp8
#SBATCH --output=logs/procap_student.%j.out
#SBATCH --error=logs/procap_student.%j.err
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p "$HOME/logs"

SCRATCH_DIR="/workdir/$USER/$SLURM_JOB_ID"
mkdir -p "$SCRATCH_DIR"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

ARCHIVE_SRC="/home2/npp8/ProCapNet/data/procap/processed/K562/distillation/distillation_dataset_k562.npz"
ARCHIVE_DST="$SCRATCH_DIR/distillation_dataset_k562.npz"
rsync -a "$ARCHIVE_SRC" "$ARCHIVE_DST"

OUTPUT_SCRATCH="$SCRATCH_DIR/output"
mkdir -p "$OUTPUT_SCRATCH"

source /home2/npp8/data/pytorch_env/bin/activate

python3 /home2/npp8/ProCapNet/src/npp8_files/train_student_fit_generator.py \
    --archive "$ARCHIVE_DST" \
    --model-save-path "$OUTPUT_SCRATCH/student.model" \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --validation-iter 100 \
    --early-stop-epochs 10 \
    --val-fraction 0.1 \
    --alpha 1.0 \
    --n-filters 512 \
    --n-layers 8 \
    --seed 42

FINAL_OUTPUT_DIR="/home2/npp8/ProCapNet/models/distilled_fit_generator"
mkdir -p "$FINAL_OUTPUT_DIR"
rsync -a "$OUTPUT_SCRATCH/" "$FINAL_OUTPUT_DIR/"
