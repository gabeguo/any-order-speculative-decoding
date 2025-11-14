#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=gpt_comparison
#
#SBATCH --time=120:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1            # Number of tasks (one per GPU)

python eval_gpt2.py \
    --n_trials 100 \
    --seq_length 512 \
    --prompt_percent 0.05 \
    --output_dir /atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison/gpt2