#!/bin/bash

# =========================
# Manual job info (to mimic SLURM vars)
# =========================
JOB_NAME="train_kvasir_student"
JOB_ID=$(date +"%Y%m%d_%H%M%S")   # fake job id using timestamp

OUT_DIR="logs_student_kvasir"
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
echo "Train Student on Kvasir dataset"
echo "==================================="

# =========================
# Run training
# =========================

python train.py --config ./config_student/training_kvasir.yaml
