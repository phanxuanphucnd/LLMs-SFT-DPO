CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --adapter_name_or_path models/AMIS/misa-vistral-7b-chat-use_unsloth-lora-4ep-lr5e5-r32-alpha16-7k7-lora \
    --template default \
    --finetuning_type lora \
    --export_dir storages/misa-vistral-7b-chat-use_unsloth-lora-4ep-lr5e5-r32-alpha16-7k7 \
    --export_size 4 \
    --export_legacy_format False 