#!/bin/bash
#SBATCH --job-name=train-languided-qata    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs_qata/%x_%j.out  # output
#SBATCH --error=logs_qata/%x_%j.err   # error

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)
echo "Test baseline with Dinnov2, save_model_qata/v9"

# load necessary modules
# module load cuda-11.4.0-gcc-11.4.0-awtybpn

# load miniconda
source /media02/tphung/third-parties/miniconda3/bin/activate vlm-Duy

# # in case above not working
# source /media02/tphung/third-parties/miniconda3/etc/profile.d/conda.sh
# conda activate [ENV_NAME]

# run code
cd /media02/tphung/workspace-Duy/LanGuideMedSeg-MICCAI2023
python train.py
python evaluate.py