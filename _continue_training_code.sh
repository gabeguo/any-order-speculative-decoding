#!/bin/bash
#
#SBATCH --reservation=asarm_scale
#SBATCH --partition=m1266
#SBATCH --account=m1266
#SBATCH --job-name=code_training
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=144:00:00
#SBATCH --nodes=1                # Single node
#SBATCH --gpus=a100:4
#SBATCH --cpus-per-task=8       # CPUs for the job
#SBATCH --ntasks=4            # Number of tasks (one per GPU)

# TODO: 
# change --output_dir
# select "openwebtext" or "bigcode/starcoderdata" dataset
# change batch_size as desired

#dataset="openwebtext"
dataset="bigcode/starcoderdata"

python -m torch.distributed.launch \
    --nproc_per_node=1 finetune_xlnet_distributed.py \
    --batch_size 24 --accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_steps 2500 --masking_warmup_steps 2500 \
    --start_masking_rate 0.85 --final_min_masking_rate 0.85 --final_max_masking_rate 0.99 \
    --items_per_epoch $((120*1000*1000)) --num_epochs 1 \
    --perm_mask_type joint \
    --output_dir $PSCRATCH/xlnet_code \
    --cache_dir $PSCRATCH/cache_sub \
    --loss_scale_type scale_by_none \
    --eval_steps 5000 \
    --save_steps 2500 \
    --eval_masking_rate 0.95 \
    --eval_num_decodes 1 \
    --wandb_project "finetune_xlnet_comparisons" \
    --packed_dataset \
    --dataset $dataset \
    --use_hf_model