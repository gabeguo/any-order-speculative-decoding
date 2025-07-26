#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=spec_decoding
#
#SBATCH --time=48:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1             # Number of tasks (one per GPU)

# TODO: change output_dir and --finetuned_model_dir
for k in 2 3 4 5 6 7 8 9 10 15 20; do
    for max_length in 128 256 512 1024; do
        output_dir="/atlas/u/gabeguo/neurips_2025/speculative_decoding/ablations/k_${k}_max_length_${max_length}"

        python -O run_decoding_eval.py \
            --finetuned_model_dir therealgabeguo/ASARM \
            --start_percentage 0.05 \
            --num_trials 500 \
            --eps 0 \
            --k $k \
            --max_length $max_length \
            --T 1 \
            --output_dir $output_dir \
            --hf_revision nlp \
            --skip_off_the_shelf \
            --skip_ngram
        python eval_perplexity.py \
            --perplexity_model "gpt2-large" \
            --batch_size 4 \
            --results_dir $output_dir \
            --skip_off_the_shelf \
            --skip_ngram
    done
done