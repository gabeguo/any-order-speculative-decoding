
dataset="openwebtext"

python -m torch.distributed.launch \
    --nproc_per_node=1 finetune_xlnet_distributed.py \
    --batch_size 12 --accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_steps 5000 --masking_warmup_steps 5000 \
    --start_masking_rate 0.90 --final_min_masking_rate 0.90 --final_max_masking_rate 0.99 \
    --items_per_epoch $((120*1000*1000)) --num_epochs 1 \
    --perm_mask_type joint \
    --cache_dir hf_cache \
    --output_dir train_nlp \
    --loss_scale_type scale_by_none \
    --eval_steps 5000 \
    --save_steps 2500 \
    --eval_masking_rate 0.95 \
    --eval_num_decodes 1 \
    --wandb_project "finetune_xlnet_comparisons" \
    --packed_dataset \
    --dataset $dataset \
    --use_hf_model