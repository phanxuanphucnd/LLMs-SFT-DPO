import pandas as pd
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the base model.
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     # llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=False,
#     bnb_4bit_compute_dtype=torch.bfl  oat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
# )

base_model_id = "storages/misa-vistral-7b-chat-use_unsloth-lora-add1ep-lr5e5-r32-alpha16-DPO"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

system_prompt = """
hiểu biết của Em:
{context}
=================
Hãy đưa ra phản hồi theo nguyên tắc sau:
1. Nếu trong CÓ thông tin liên quan trong `hiểu biết của Em`, hãy trả lời ngắn gọn trong khoảng 3-5 câu theo kiểu hướng dẫn theo từng bước dễ hiểu. Mỗi câu 1 dòng.
2. Không được phép trả lời quá dài, vì người dùng sẽ không đọc hết.
3. Xưng hô với người dùng là 'Quý khách'. KHÔNG ĐƯỢC PHÉP gọi người dùng là 'bạn'. KHÔNG ĐƯỢC PHÉP tự xưng là 'tôi'.
4. Nếu trong KHÔNG CÓ thông tin cho câu trả lời, hãy giải thích cho người dùng biết rằng chưa hiểu câu hỏi và bảo người dùng đặt lại câu hỏi.
5. Không được trùng lặp thông tin từ bước trước.
"""

df = pd.read_excel('data/testset/recors4+vistral-eval.xlsx')

# df = pd.read_csv('/home/ec2-user/namph/meinvoice.csv')

questions = df['question'].values.tolist()
context = df['context'].values.tolist()
answers = []
inferrence_time = []

for i in tqdm(range(len(questions))):
    q = questions[i]
    c = context[i]
    start_time = time.time()
    conversation = [{"role": "system", "content": system_prompt.format(context="<knowledge>\n" + c +"\n<knowledge/>") }]
    conversation.append({"role": "user", "content": q })
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
    print("Assistant: ", assistant)
    x = time.time() - start_time
    throughput = out_ids[:, model_input.size(1): ].shape[1]/x
    print("--- %s seconds ---" % (x), throughput)
    answers.append(assistant)
    inferrence_time.append(throughput)
    
new_df = pd.DataFrame({'question': questions, 'context': context, 'answer': answers, "inferrence_time": inferrence_time})

new_df.to_csv('phuc_px_test_answer-dpo.csv', index=False)