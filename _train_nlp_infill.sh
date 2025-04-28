#!/bin/bash
#
#SBATCH --reservation=asarm_scale
#SBATCH --partition=m1266
#SBATCH --account=m1266
#SBATCH --job-name=nlp_infill
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

export HF_HOME=$SCRATCH/cache/huggingface
echo $HF_HOME

dataset="openwebtext"

python -m torch.distributed.launch \
    --nproc_per_node=4 finetune_xlnet_distributed.py \
    --batch_size 24 --accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_steps 2500 --masking_warmup_steps 2500 \
    --start_masking_rate 0.15 --final_min_masking_rate 0.60 --final_max_masking_rate 0.75 \
    --items_per_epoch $((120*1000*1000)) --num_epochs 1 \
    --perm_mask_type joint \
    --output_dir $PSCRATCH/xlnet_nlp_infill \
    --cache_dir $HF_HOME \
    --loss_scale_type scale_by_none \
    --eval_steps 5000 \
    --save_steps 2500 \
    --eval_masking_rate 0.6 \
    --eval_num_decodes 1 \
    --wandb_project "finetune_xlnet_comparisons" \
    --packed_dataset \
    --dataset $dataset