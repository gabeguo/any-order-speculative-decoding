#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=code_eval
#
#SBATCH --time=24:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1             # Number of tasks (one per GPU)

python eval_coding.py \
    --finetuned_model_dir "/atlas/u/gabeguo/finetune_xlnet/03_14_25/learn_to_code/2025-03-14_124247/checkpoint-75000" \
    --output_dir "/atlas/u/gabeguo/dummy_runs/coding" \
    --max_tasks 2000