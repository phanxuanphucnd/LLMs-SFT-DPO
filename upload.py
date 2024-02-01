from huggingface_hub import HfApi, login, create_repo

login("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")


api = HfApi()

repo_id = "phucpx247/misa-vistral-7b-chat-use_unsloth-lora-1ep-lr5e5-r32-alpha16-DPO-adapter"

create_repo(repo_id, private=True)


api.upload_folder(
    folder_path="models/DPO/misa-vistral-7b-chat-use_unsloth-lora-1ep-lr5e5-r32-alpha16-DPO",
    repo_id=repo_id,
    repo_type="model",
)