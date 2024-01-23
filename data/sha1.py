import hashlib

with open('data/misa_amiskt_gpt4_data_vi.json', "rb") as f:
    sha1 = hashlib.sha1(f.read()).hexdigest()
    print(sha1)