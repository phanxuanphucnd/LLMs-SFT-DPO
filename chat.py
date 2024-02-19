import pandas as pd
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import json

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model_id = "storages/vistral-7b-chat-public-dataset-DPO"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
eval_tokenizer.pad_token = eval_tokenizer.eos_token



# while True: 

with open('data/comparision_public_data_vi.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)


for item in json_data:
    q = item['instruction']
    # q = input("Please enter a question:\n")
    # system_prompt = input("Please enter a system prompt:\n")
    system_prompt = item['system']

    start_time = time.time()

    conversation = [{"role": "system", "content": system_prompt}]
    conversation.append({"role": "user", "content": q})
    runtimeFlag = "cuda:0"
    model_input = eval_tokenizer.apply_chat_template(conversation, return_tensors="pt").to(runtimeFlag)
    
    out_ids = base_model.generate(
        input_ids=model_input,
        max_new_tokens=768,
        do_sample=True,
        top_p=1,
        top_k=40,
        temperature=0.1,
        repetition_penalty=1.05)
    
    assistant = eval_tokenizer.batch_decode(out_ids[:, model_input.size(1): ], skip_special_tokens=True)[0].strip()

    print("\nAssistant: ", assistant)

    x = time.time() - start_time

    throughput = out_ids[:, model_input.size(1): ].shape[1]/x

    print("--- %s seconds ---" % (x), throughput)

    print("\n************************************")
