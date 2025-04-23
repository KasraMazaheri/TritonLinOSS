#!/bin/bash
#SBATCH --job-name=ssm-training-repeats
#SBATCH --account=drl
#SBATCH --partition=drl
#SBATCH --qos=drl-main
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch/jaredb/job_output_%j.log
#SBATCH --error=/data/scratch/jaredb/job_error_%j.log
#SBATCH --nodes=1
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source /data/scratch/jaredb/miniconda3/etc/profile.d/conda.sh
conda activate linoss

python /data/scratch/jaredb/linoss/scripts/run/run_experiment.py --model_names Transformer --dataset_names ScoopingPowder --config_folder config/spread --task_id ${SLURM_ARRAY_TASK_ID} --num_tasks 8 --save_model
