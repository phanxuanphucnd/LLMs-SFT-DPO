import json

with open('data/alpaca_gpt4_data_en.json', 'r', encoding='utf-8') as f:
    data = json.load(f)



for item in data:
    # item['system'] = "YOU ARE A MISA ASSISTANT."
    item['context'] = "CONTEXT"


with open('data/alpaca_gpt4_data_en.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
