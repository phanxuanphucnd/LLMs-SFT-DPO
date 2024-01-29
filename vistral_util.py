from typing import List, Dict, Text

import re
import time
import json
import yaml
import copy
import logging
import requests
from prompt_template import PROMPT_TEMPLATE_CHITCHAT, PROMPT_EXPLAIN, PROMPT_TEMPLATE, EXTRACT_SLOT_VALUE_TEMPLATE

logger = logging.getLogger("vistral")

with open("configs/vistral_config.yml", "r") as f:
    vistral_config = yaml.safe_load(f)

class VistralGenerator():
    def __init__(self) -> None:
        self.default_url = vistral_config["MISA-Vistral-7B-01"]["url"]
        self.default_api_key = vistral_config["MISA-Vistral-7B-01"]["api-key"]

    @staticmethod
    def get_model_connection(model: str):
        return vistral_config.get(model, {})
    
    def call_vistral(self, messages: List[Dict[Text, Text]], max_tokens = 768, model: str = "", **kwargs):
        connection_info = self.get_model_connection(model)
        self.url = connection_info.get("url", self.default_url) + "/get_answer"
        self.headers = {
            "api-key": connection_info.get("api-key", self.default_api_key),
        }
        
        logger.debug(f"Send messsage to Vistral:{messages}")
        json_data = {
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs
        }
        res = requests.post(self.url, headers = self.headers, json = json_data, stream = True, verify = False)
        if res.status_code == 200:
            return res
        raise Exception(f"Vistral error: {res.text}")
        
    def get_answer_chitchat(self, question: Text, model: str):
        """
         Get answer by intent. This method is used to get answer by intent.

         Args:
             question: User question
             his_data: List of messages to send
             stream: if False return answer as string
        """
        messages = []
        messages.append({"role": "system", "content": PROMPT_TEMPLATE_CHITCHAT})
        messages.append({"role": "user", "content": question})

        res = self.call_vistral(
            messages=messages,
            model=model,
            stream=True,
        )
        for r in res.iter_content(64):
            yield json.loads(r.decode("utf-8"))["token"]
        
    def get_answer_explain(self, question: Text, model: str):
        """
         Get OpenAI answer explain.

         Args:
             question: User question
             his_data: List of dicts that are passed to openai
             stream: if False return answer as string
        """
        messages = []
        messages.append({"role": "system", "content": PROMPT_EXPLAIN})
        messages.append({"role": "user", "content": question})

        res = self.call_vistral(
            messages=messages,
            model=model,
            stream=True,
        )
        for r in res:
            yield json.loads(r.decode("utf-8"))["token"]
    
    def get_answer(
        self,
        question: Text,
        doc: Text,
        model: str
    ):
        """
         Get answer to a question.

         Args:
            :param question: User question
            :param candidates: list of documents relevant to this question
            :param stream: if False return answer as string
        """
        messages = []

        doc_str = "<knowledge>\n" + doc + "\n</knowledge>"
        logger.debug(f"Related docs for question: {question}\n")
        prompt_content = PROMPT_TEMPLATE.format(
            context=doc_str)
        
        messages.append({"role": "system", "content": prompt_content})
        messages.append({"role": "user", "content": question})

        res = self.call_vistral(
            messages=messages,
            model=model,
            stream=True,
        )
        for r in res:
            yield json.loads(r.decode("utf-8"))["token"]
    
    def capitalize_after_punctuation(self, text):
        # Define a regular expression pattern to match words following punctuation
        pattern = r'(?<=[.!?])\s*\w'
        
        # Define a function to capitalize the matched word
        def capitalize(match):
            return match.group().upper()
        
        # Use the re.sub() function to replace the matched words with their uppercase versions
        result = re.sub(pattern, capitalize, text)
    
        return result
    def clean_answer(self, answer: Text):
        if "](" in answer:
            answer = answer.replace("[link](", "")
            answer = answer.replace("](", ": ")
            answer = answer.replace(" [", " ")
            answer = answer.replace(")", "")
        # if "Câu trả lời:" in answer:
        #     answer = answer.split("Câu trả lời:")[-1]
        # if "Trả lời:" in answer:
        #     answer = answer.split("Trả lời:")[-1]
        # if "Answer:" in answer:
        #     answer = answer.split("Answer:")[-1]
        # if "phản hồi:" in answer:
        #     answer = answer.split("phản hồi:")[-1]
        # if "Phản hồi:" in answer:
        #     answer = answer.split("Phản hồi:")[-1]
        sentences = answer.split(". ")
        sentences = [sentence for sentence in sentences if "http" not in sentence]
        answer = ". ".join(sentences)
        answer = answer.replace("quý khách", "Quý khách")
        if answer.startswith("Quý khách"):
            answer = answer.replace("Quý khách ", "", 1)
            answer = answer[0].upper() + answer[1:]
        answer = self.capitalize_after_punctuation(answer)
        return answer.strip()

if __name__ == "__main__":
    vistral_gen = VistralGenerator()
    gen = vistral_gen.get_answer_chitchat("bạn có thể làm gì")
    for i in gen:
        print(i, end = "")
