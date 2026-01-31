#!/bin/bash

# =========================
# Manual job info (to mimic SLURM vars)
# =========================
JOB_NAME="train_mosmed_baseline_kd"
JOB_ID=$(date +"%Y%m%d_%H%M%S")   # fake job id using timestamp

OUT_DIR="logs_baseline_kd_mosmed"
mkdir -p ${OUT_DIR}

OUT_FILE="${OUT_DIR}/${JOB_NAME}_${JOB_ID}.out"
ERR_FILE="${OUT_DIR}/${JOB_NAME}_${JOB_ID}.err"

# =========================
# Redirect stdout & stderr
# =========================
exec >"${OUT_FILE}" 2>"${ERR_FILE}"

# =========================
# Logging (same spirit as SLURM)
# =========================
echo "Job name: ${JOB_NAME}"
echo "Job ID: ${JOB_ID}"
echo "Executing on the machine: $(hostname)"
echo "Train Baseline KD with 1 * Multiple Temperature Logit [1, 2, 3, 4, 5] + 0 * Importance Map Logit MSE p1 p2 [refined_os32, refined_os16, refined_os8, os4] on MosMedPlus dataset"
echo "==================================="

# =========================
# Run training
# =========================

python train.py --config ./config_baseline_kd/training_mosmed.yaml
