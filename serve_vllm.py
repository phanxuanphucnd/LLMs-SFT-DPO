from vllm import LLM
llm = LLM("/home/ec2-user/LLMEvaluator/abc/llms/storages/misa-vistral-7b-dpo-use_unsloth-lora-4ep-lr5e5-r32-alpha64", tensor_parallel_size=1)

messages = [{"role": "system", "content": "bạn là misa chatbot"}, {"role": "user", "content": "alo"}]
output = llm.generate(messages)
print(output)
