import os

os.environ['HUGGINGFACEHUB_API_TOKEN']="<Your API Token>"

# use below 5 lines in case you need to change for default haggingface path in your system

#PATH = '<path to change>' like /user/drive/NEW_MODEL_CACHE

#os.environ['TRANSFORMERS_CACHE'] = '<path to change>'  like /user/drive/NEW_MODEL_CACHE/models

#os.environ['HF_HOME'] = PATH

#os.environ['HF_DATASETS_CACHE'] = '<path to change>' like /user/drive/NEW_MODEL_CACHE/datasets

#os.environ['TORCH_HOME'] = '<path to change>' like /user/drive/NEW_MODEL_CACHE/

from huggingface_hub import snapshot_download

from transformers import AutoModelForCausalLM,AutoTokenizer

from pathlib import Path


model_name="mistralai/Mistral-7B-v0.3"

tokenizer= AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained(f"<your path>/mistralmodel/tokenizer/{model_name}")

 

model=AutoModelForCausalLM.from_pretrained(model_name)

model.save_pretrained(f"<your path>/mistralmodel/model/{model_name}")
