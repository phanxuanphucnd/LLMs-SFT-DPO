from typing import List, Dict, Any
import torch
from threading import Thread
import time
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
    TextIteratorStreamer,
    pipeline
)
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16
# )

device = torch.device("cuda:0")

base_model_id = "/home/ec2-user/LLMEvaluator/abc/llms/storages/misa-vistral-7b-dpo-use_unsloth-lora-4ep-lr5e5-r32-alpha64"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    # quantization_config=bnb_config,
    device_map="auto"
)
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def answer(
    messages: List[Dict[str, str]],
    max_tokens: int = 768,
    do_sample: bool = True,
    top_p: float = 1.0,
    top_k: int = 40,
    temperature: float = 0.1,
    repetition_penalty: float = 1.05,
    **kwargs
):
    model_input = eval_tokenizer.apply_chat_template(messages, return_tensors = "pt").to(device)
    
    generate_kwargs = dict(
        input_ids = model_input,
        max_new_tokens = max_tokens,
        do_sample = do_sample,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
        repetition_penalty = repetition_penalty,
    )
    out_ids = base_model.generate(**generate_kwargs)
    answer = eval_tokenizer.batch_decode(out_ids[:, model_input.size(1): ], skip_special_tokens=True)[0].strip()
    return answer

def answer_stream(
    messages: List[Dict[str, str]],
    max_tokens: int = 768,
    do_sample: bool = True,
    top_p: float = 1.0,
    top_k: int = 40,
    temperature: float = 0.1,
    repetition_penalty: float = 1.05,
    **kwargs
):
    model_input = eval_tokenizer.apply_chat_template(messages, return_tensors = "pt").to(device)
    streamer = TextIteratorStreamer(eval_tokenizer, skip_prompt = True, skip_special_tokens = True)
    
    generate_kwargs = dict(
        input_ids = model_input,
        streamer = streamer,
        max_new_tokens = max_tokens,
        do_sample = do_sample,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
        repetition_penalty = repetition_penalty,
    )
    t = Thread(target=base_model.generate, kwargs=generate_kwargs)
    t.start()

    for token in streamer:
        yield token

app = FastAPI()

@app.get("/healthz")
def get_healthz():
    return {"status": "ok"}

class GetAnswerRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 768
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 40
    temperature: float = 0.1
    repetition_penalty: float = 1.05
    stream: bool = True
    kwargs_add_more: Dict[Any] = {}

def render_stream(response):
    for i, token in enumerate(response):
        yield json.dumps({"idx": i, "token": token})

@app.post("get_answer")
def get_answer(request_body: GetAnswerRequest):
    payload = request_body.__dict__
    is_stream = payload.pop("stream")
    kwargs_add_more = payload.pop("kwargs_add_more")
    
    if is_stream:
        return StreamingResponse(render_stream(answer_stream(**payload, **kwargs_add_more)))
    else:
        return {
            "idx": 0,
            "token": answer(**payload, **kwargs_add_more)
        }
    

# conversation = [{"role": "system", "content": "bạn là misa chatbot"}, {"role": "user", "content": "alo"}]

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "serve:app",
        host = "0.0.0.0",
        port = 8789
    )

