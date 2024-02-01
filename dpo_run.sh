# CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus 2 --master_port=9902 src/train_bash.py \
#     --deepspeed configs/ds_dpo_config.json \

CUDA_VISIBLE_DEVICES=2 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --adapter_name_or_path models/AMIS/misa-vistral-7b-chat-use_unsloth-lora-4ep-lr5e5-r32-alpha16 \
    --create_new_adapter \
    --dataset comparision_makt_labeled_3k_vi \
    --template 'llama2' \
    --finetuning_type lora \
    --lora_target all \
    --output_dir models/DPO/misa-vistral-7b-chat-use_unsloth-lora-1ep-lr5e5-r32-alpha16-DPO \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --fp16 \
    --use_unsloth