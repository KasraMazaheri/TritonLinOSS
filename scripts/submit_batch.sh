#!/bin/bash

source activate linoss

# Adjust these configs
model_names=('LinOSS')
dataset_names=('EigenWorms' 'SelfRegulationSCP1' 'SelfRegulationSCP2' 'EthanolConcentration' 'Heartbeat' 'MotorImagery')
config_folder='config/grid'

# Batching
task_id=$LLSUB_RANK
num_tasks=$LLSUB_SIZE

python scripts/run/run_experiment.py \
    --model_names "${model_names[@]}" \
    --dataset_names "${dataset_names[@]}" \
    --config_folder "$config_folder" \
    --task_id $task_id \
    --num_tasks $num_tasks