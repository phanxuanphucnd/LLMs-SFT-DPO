import argparse
import torch
import peft
import json
import shutil
from peft.utils import _get_submodules
import os
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig #CodeLlamaTokenizer
import gc
import copy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--peft", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()

def dequantize_model(model, tokenizer, to, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    if os.path.exists(to):
        return AutoModelForCausalLM.from_pretrained(to, torch_dtype=torch.bfloat16, device_map="auto")

    os.makedirs(to, exist_ok=True)

    cls = bnb.nn.Linear4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype
                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)
                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        model.is_loaded_in_4bit = False
        print("Saving dequantized model...")

        model.save_pretrained(to)

        tokenizer.save_pretrained(to)
        config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
        config_data.pop("quantization_config", None)
        config_data.pop("pretraining_tp", None)

        with open(os.path.join(to, 'config.json'), 'w') as config:
            config.write(json.dumps(config_data, indent=2))
        return model


def main():
    model_path = "jan-hq/Vistral-7B-Chat-DPO"
    adapter_path = "models/output/misa-vistral-7b-dpo-use_unsloth-lora-4ep-lr5e5-r32-alpha16"
    out_path = "storages/misa-vistral-7b-dpo-use_unsloth-lora-4ep-lr5e5-r32-alpha16"
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"Loading base model: {model_path}")
    model = None
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if os.path.exists(f"{model_path}-dequantized"):
        model = AutoModelForCausalLM.from_pretrained(
            f"{model_path}-dequantized",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = dequantize_model(model, tokenizer, to=f"{model_path}-dequantized")

    model = PeftModel.from_pretrained(model=model, model_id=adapter_path)

    model = model.merge_and_unload()

    print("Successfully loaded and merged model, saving...")

    model.save_pretrained(out_path, safe_serialization=True, max_shard_size='4GB')

    tokenizer.save_pretrained(out_path)

    config_data = json.loads(open(os.path.join(out_path, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(out_path, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))

    print(f"Model saved: {out_path}")
    # if args.push:
    #     print(f"Saving to hub ...")
    #     model.push_to_hub(out_path, use_temp_dir=False)
    #     tokenizer.push_to_hub(out_path, use_temp_dir=False)
    #     print("Model successfully pushed to hf.")

if __name__ == "__main__":
    main()