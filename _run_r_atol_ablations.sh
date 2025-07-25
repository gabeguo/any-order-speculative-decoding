#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=r_atol_ablations
#
#SBATCH --time=48:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1             # Number of tasks (one per GPU)

# TODO: change output_dir and --finetuned_model_dir
k=5
max_length=256
for r_atol in 0 1e-4 1e-3 1e-2 5e-2 1e-1; do
    output_dir="/atlas/u/gabeguo/neurips_2025/speculative_decoding/ablations/r_atol_${r_atol}"

    python -O run_decoding_eval.py \
        --finetuned_model_dir therealgabeguo/ASARM \
        --start_percentage 0.05 \
        --num_trials 500 \
        --eps 0 \
        --k $k \
        --max_length $max_length \
        --T 1 \
        --output_dir $output_dir \
        --hf_revision nlp_210k_plus_75k \
        --skip_off_the_shelf \
        --skip_ngram \
        --r_atol $r_atol
    python eval_perplexity.py \
        --perplexity_model "gpt2-large" \
        --batch_size 4 \
        --results_dir $output_dir \
        --skip_off_the_shelf \
        --skip_ngram
done