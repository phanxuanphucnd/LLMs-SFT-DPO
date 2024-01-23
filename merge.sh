python src/export_model.py \
    --model_name_or_path jan-hq/Vistral-7B-Chat-DPO \
    --adapter_name_or_path models/misa_amiskt/v1.0/Vistral-7B \
    --template 'llama2' \
    --finetuning_type lora \
    --export_dir './storages/' \
    --export_size 2 \
    --export_legacy_format False