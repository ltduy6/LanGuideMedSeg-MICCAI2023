#!/bin/bash

# =========================
# Manual job info (to mimic SLURM vars)
# =========================
JOB_NAME="train_qata_baseline_kd"
JOB_ID=$(date +"%Y%m%d_%H%M%S")   # fake job id using timestamp

OUT_DIR="logs_baseline_kd_qata"
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
echo "Train Baseline KD with 1 * Multiple Temperature Logit [1] + 0 * Feature Distill MSE p1 p2 [os32, os16, os8, os4] on Qata dataset"
echo "==================================="

# =========================
# Run training
# =========================

python train.py --config ./config_baseline_kd/training_qata.yaml --temps '[1.0]'

echo "Train Baseline KD with 1 * Multiple Temperature Logit [2] + 0 * Feature Distill MSE p1 p2 [os32, os16, os8, os4] on Qata dataset"
echo "==================================="

python train.py --config ./config_baseline_kd/training_qata.yaml --temps '[2.0]'

echo "Train Baseline KD with 1 * Multiple Temperature Logit [3] + 0 * Feature Distill MSE p1 p2 [os32, os16, os8, os4] on Qata dataset"
echo "==================================="

python train.py --config ./config_baseline_kd/training_qata.yaml --temps '[3.0]'

echo "Train Baseline KD with 1 * Multiple Temperature Logit [4] + 0 * Feature Distill MSE p1 p2 [os32, os16, os8, os4] on Qata dataset"
echo "==================================="

python train.py --config ./config_baseline_kd/training_qata.yaml --temps '[4.0]'

echo "Train Baseline KD with 1 * Multiple Temperature Logit [5] + 0 * Feature Distill MSE p1 p2 [os32, os16, os8, os4] on Qata dataset"
echo "==================================="

python train.py --config ./config_baseline_kd/training_qata.yaml --temps '[5.0]'

echo "Train Baseline KD with 1 * Multiple Temperature Logit [6] + 0 * Feature Distill MSE p1 p2 [os32, os16, os8, os4] on Qata dataset"
echo "==================================="

python train.py --config ./config_baseline_kd/training_qata.yaml --temps '[6.0]'
echo "Training completed."
