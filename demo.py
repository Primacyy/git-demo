import torch
import json
import pandas as pd
from datasets import load_dataset,Dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig 
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "C:/Users/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/deepseek-r1-1.5b"
# tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True).to("cuda")

print("---------模型加载成功!------------")
## 处理数据集
data_path = ".\medical_multi_data.json"
data = pd.read_json(data_path)
processed_data = Dataset.from_pandas(data)
# print(processed_data[0])
# print(type(processed_data[0]))

def format_conversion(processed_data):
    formatted_texts = []
    for example in processed_data["conversation"]:
        formatted_text = ""
        for turn in example:
            if turn.get("system"):
                formatted_text += f"[System] {turn['system']}\n"
            formatted_text += f"[input] {turn['input']}\n"
            formatted_text += f"[output] {turn['output']}\n"
        formatted_texts.append(formatted_text)
    return {"conversation": formatted_texts}
        #     formatted_text += f"{dict(system = turn['system'])}\n"
        # formatted_text += f"{dict(input = turn['input'])}\n"
        # formatted_text += f"{dict(output = turn['output'])}\n"
final_data = processed_data.map(format_conversion,batched=True)
def w_data(d):
    with open("mydata.json",'w',encoding="utf-8") as f:
        for s in d["conversation"]:
            json_line = json.dumps(dict(conversation=s),ensure_ascii=False)
            f.write(json_line+"\n")
final_data.map(w_data,batched=True)
# print(len(data))
# dataset = load_dataset(path="json",data_files="medical_multi_data.json",split="train")
# dataset_train_test = dataset.train_test_split(test_size=0.1)
# data_train = dataset_train_test["train"]
# data_test = dataset_train_test["test"]
# print(f"train size:{ len(data_train) }")
print("---------数据准备完成!------------")

## 编写tokenizer处理工具
# def tokenizer_function(example):

