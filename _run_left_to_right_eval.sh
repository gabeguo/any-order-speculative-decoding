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
num_trials=1
T=1

for task in "lambada" "hellaswag"; do
    output_dir="/atlas/u/gabeguo/neurips_2025/rebuttal/${task}"
    output_dir_finetuned="${output_dir}/xlnet_finetuned"
    python eval_infilling.py \
        --max_items $max_items \
        --num_trials $num_trials \
        --finetuned_model_dir therealgabeguo/ASARM \
        --output_dir $output_dir_finetuned \
        --T $T \
        --dataset $task

    # output_dir_off_the_shelf="${output_dir}/xlnet_off_the_shelf"
    # python eval_infilling.py $prompting \
    #     --max_items $max_items \
    #     --num_trials $num_trials \
    #     --output_dir $output_dir_off_the_shelf \
    #     --T $T \
    #     --hf_revision $hf_revision
done