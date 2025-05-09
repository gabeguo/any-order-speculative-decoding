#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=285k_iter_speculative_decoding
#
#SBATCH --time=48:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1             # Number of tasks (one per GPU)

# TODO: change output_dir and --finetuned_model_dir
output_dir="/atlas/u/gabeguo/neurips_2025/speculative_decoding/_285k_iter"

python -O run_decoding_eval.py \
    --finetuned_model_dir therealgabeguo/ASARM \
    --start_percentage 0.05 \
    --num_trials 700 \
    --eps 0 \
    --k 5 \
    --T 1 \
    --output_dir $output_dir \
    --hf_revision nlp_210k_plus_75k \
    --skip_off_the_shelf
python eval_perplexity.py \
    --perplexity_model "gpt2-large" \
    --batch_size 4 \
    --results_dir $output_dir \
    --skip_off_the_shelf