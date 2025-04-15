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

# TODO: change output_dir
# TODO: Download the evaluation dataset from https://github.com/HKUNLP/DiffuLLaMA/blob/main/evaluation/evaluation/cloze_test_val__spring2016.csv
# and put it in the eval_datasets folder

max_items=2000
T=1

for prompting in "--short_prompt" ""; do
    output_dir="/atlas/u/gabeguo/dummy_runs/nlp_infill/prompting_${prompting}"
    python eval_infilling.py $prompting \
        --max_items $max_items \
        --finetuned_model_dir therealgabeguo/ASARM \
        --output_dir $output_dir \
        --T $T
done