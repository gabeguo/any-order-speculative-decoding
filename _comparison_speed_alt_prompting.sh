#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=asarm_speed_comparison
#
#SBATCH --time=48:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1             # Number of tasks (one per GPU)

# TODO: change output_dir and --finetuned_model_dir
output_dir="/atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison_redo/asarm_left_to_right_match_length"

python -O run_decoding_eval.py \
    --finetuned_model_dir therealgabeguo/ASARM \
    --max_length 570 \
    --start_percentage 0.05 \
    --num_trials 200 \
    --eps 0 \
    --k 5 \
    --T 1 \
    --output_dir $output_dir \
    --hf_revision nlp \
    --skip_off_the_shelf \
    --left_to_right \
    --alt_prompt_loading
python eval_perplexity.py \
    --perplexity_model "gpt2-large" \
    --batch_size 4 \
    --results_dir $output_dir \
    --skip_off_the_shelf