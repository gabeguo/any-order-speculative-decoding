#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=nlp_infill
#
#SBATCH --time=24:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1             # Number of tasks (one per GPU)

# TODO: change output_dir and --finetuned_model_dir
# TODO: Download the evaluation dataset from https://github.com/HKUNLP/DiffuLLaMA/blob/main/evaluation/evaluation/cloze_test_val__spring2016.csv
# and put it in the eval_datasets folder

max_items=1000
T=1

for prompting in "--short_prompt" ""; do
    output_dir="/atlas/u/gabeguo/dummy_runs/nlp_infill/prompting_${prompting}"
    python eval_infilling.py $prompting \
        --max_items $max_items \
        --finetuned_model_dir /atlas/u/gabeguo/finetune_xlnet/03_08_25/narrow_mask_rate_range/0_9_to_0_99/2025-03-10_122308/checkpoint-75000 \
        --output_dir $output_dir \
        --T $T
done