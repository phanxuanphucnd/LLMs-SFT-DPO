from huggingface_hub import HfApi, login

login(token = "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")

api = HfApi()

api.upload_folder(
    folder_path="/home/ec2-user/LLMEvaluator/abc/llms/storages/misa-vistral-7b-dpo-use_unsloth-lora-3ep-lr5e5-r32-alpha16slow",
    repo_id="ShiniChien/misa-vistral-7b-dpo-use_unsloth-lora-3ep-lr5e5-r32-alpha16slow",
    repo_type="model",
)