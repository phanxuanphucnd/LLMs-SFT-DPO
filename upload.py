from huggingface_hub import HfApi, login, create_repo

login("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")


api = HfApi()

repo_id = "phucpx247/misa-vistral-7b-chat-lora-4ep-lr5e5-r32-alpha16-11k1"

create_repo(repo_id, private=True)


api.upload_folder(
    folder_path="storages/misa-vistral-7b-chat-lora-4ep-lr5e5-r32-alpha16-11k1",
    repo_id=repo_id,
    repo_type="model",
)
