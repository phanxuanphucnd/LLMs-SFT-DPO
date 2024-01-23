from enum import Enum
from collections import defaultdict, OrderedDict
from typing import Dict, Optional


CHOICES = ["A", "B", "C", "D"]

DEFAULT_MODULE = defaultdict(str)

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text"
}

IGNORE_INDEX = -100

LAYERNORM_NAMES = {"norm", "ln"}

LOG_FILE_NAME = "trainer_log.jsonl"

METHODS = ["full", "freeze", "lora"]

PEFT_METHODS = ["lora"]

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft"
}

class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"


def register_model_group(
    models: Dict[str, Dict[DownloadSource, str]],
    module: Optional[str] = None,
    template: Optional[str] = None
) -> None:
    prefix = None
    for name, path in models.items():
        if prefix is None:
            prefix = name.split("-")[0]
        else:
            assert prefix == name.split("-")[0], "prefix should be identical."
        SUPPORTED_MODELS[name] = path
    if module is not None:
        DEFAULT_MODULE[prefix] = module
    if template is not None:
        DEFAULT_TEMPLATE[prefix] = template


register_model_group(
    models={
        "LLaMA-7B": {
            DownloadSource.DEFAULT: "huggyllama/llama-7b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-7b"
        },
        "LLaMA-13B": {
            DownloadSource.DEFAULT: "huggyllama/llama-13b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-13b"
        },
        "LLaMA-30B": {
            DownloadSource.DEFAULT: "huggyllama/llama-30b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-30b"
        },
        "LLaMA-65B": {
            DownloadSource.DEFAULT: "huggyllama/llama-65b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-65b"
        }
    }
)


register_model_group(
    models={
        "LLaMA2-7B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-ms"
        },
        "LLaMA2-13B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-ms"
        },
        "LLaMA2-70B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-ms"
        },
        "LLaMA2-7B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-chat-ms"
        },
        "LLaMA2-13B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-chat-ms"
        },
        "LLaMA2-70B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-chat-ms"
        }
    },
    template="llama2"
)


register_model_group(
    models={
        "Mistral-7B": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-v0.1"
        },
        "Mistral-7B-Chat": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.1"
        },
        "Mistral-7B-v0.2-Chat": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.2"
        }
    },
    template="mistral"
)


register_model_group(
    models={
        "Mixtral-8x7B": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-v0.1"
        },
        "Mixtral-8x7B-Chat": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1"
        }
    },
    template="mistral"
)


register_model_group(
    models={
        "Phi-1.5-1.3B": {
            DownloadSource.DEFAULT: "microsoft/phi-1_5",
            DownloadSource.MODELSCOPE: "allspace/PHI_1-5"
        },
        "Phi-2-2.7B": {
            DownloadSource.DEFAULT: "microsoft/phi-2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/phi-2"
        }
    },
    module="Wqkv"
)


register_model_group(
    models={
        "Zephyr-7B-Alpha-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-7b-alpha",
            DownloadSource.MODELSCOPE: "AI-ModelScope/zephyr-7b-alpha"
        },
        "Zephyr-7B-Beta-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-7b-beta",
            DownloadSource.MODELSCOPE: "modelscope/zephyr-7b-beta"
        }
    },
    template="zephyr"
)
