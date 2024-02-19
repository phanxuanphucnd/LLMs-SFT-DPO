CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --adapter_name_or_path models/vistral-7b-chat-public-dataset-DPO \
    --template default \
    --finetuning_type lora \
    --export_dir storages/vistral-7b-chat-public-dataset-DPO \
    --export_size 4 \
    --export_legacy_format False 