# MSG：很容易爆显存

batch=2
gradient_accumulation_steps=4
epoch=3
learning_rate=1.4e-4
card_type="RTX3090"
card_number=8
cuda_visible_devices=0,1,2,3,4,5,6,7
training_port=20001
dataset_name="/nfs-shared/data/JARVIS/out/dataset/mc-qa-3_12_25-277k"
version="mc-llama_3.1_8b_instruct-lora-mcqa_v3_12_25_277k-9_5"  # {model_version}_{dataset_version}_{training_date}
WANDB_NAME="$version-$card_type-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"

export WANDB_MODE="offline"
export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=MARKOV
export WANDB_NOTES="[24-11-06] 18668 vision language convs (instruction state action)"

deepspeed --include localhost:$cuda_visible_devices --master_port=$training_port train/sft.py \
    --dataset_name $dataset_name \
    --model_name_or_path "/nfs-shared/models/llama-3.1/llama-3.1-8b-instruct" \
    --report_to "wandb" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --output_dir "/nfs-shared-2/zhwang/JARVIS/checkpoints/$WANDB_NAME" \
    --run_name $WANDB_NAME \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --bf16 True \
    --remove_unused_columns False \
    --max_seq_length 1024 \
    --use_peft True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_target_modules "all-linear" \
    --deepspeed train/deepspeed_config_s2.json 

