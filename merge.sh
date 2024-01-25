python src/export_model.py \
    --model_name_or_path jan-hq/Vistral-7B-Chat-DPO \
    --adapter_name_or_path models/output/misa-vistral-7b-dpo-use_unsloth-lora-4ep-lr5e5-r32-alpha16 \
    --template default \
    --finetuning_type lora \
    --export_dir storages/misa-vistral-7b-dpo-use_unsloth-lora-3ep-lr5e5-r32-alpha16slow \
    --export_size 4 \
    --export_legacy_format False