from huggingface_hub import HfApi, login, create_repo

login("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")


api = HfApi()

repo_id = "phucpx247/vnchatbot-vistral-7b-chat-use_unsloth-lora-5ep-lr5e5-r32-alpha16-sft"

create_repo(repo_id, private=True)


api.upload_folder(
    folder_path="models/AFIS/vnchatbot-vistral-7b-chat-lora-5ep-lr5e5-r32-alpha16",
    repo_id=repo_id,
    repo_type="model",
)