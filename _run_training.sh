#!/bin/bash
#
#SBATCH --partition=atlas
#SBATCH --account=atlas
#SBATCH --job-name=training
#
#SBATCH --time=120:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a4000ada:1
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=1            # Number of tasks (one per GPU)

# TODO: 
# change --output_dir
# select "openwebtext" or "bigcode/starcoderdata" dataset
# change batch_size as desired

#dataset="openwebtext"
dataset="bigcode/starcoderdata"

python -m torch.distributed.launch \
    --nproc_per_node=1 finetune_xlnet_distributed.py \
    --batch_size 8 --accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_steps 5000 --masking_warmup_steps 5000 \
    --start_masking_rate 0.15 --final_min_masking_rate 0.85 --final_max_masking_rate 0.99 \
    --items_per_epoch $((24*1000*1000)) --num_epochs 1 \
    --perm_mask_type joint \
    --output_dir /atlas/u/gabeguo/dummy_runs/training \
    --loss_scale_type scale_by_none \
    --eval_steps 500 \
    --eval_masking_rate 0.95 \
    --eval_num_decodes 2 \
    --wandb_project "finetune_xlnet_comparisons" \
    --packed_dataset \
    --dataset $dataset