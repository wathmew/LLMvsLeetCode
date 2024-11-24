import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN") # make a .env for this and put your access token as HF_TOKEN=whateverYourAccessTokenIs

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
#print("GPU available ", torch.cuda.is_available())

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

# look at the ipnyb from now on, won't be putting anymore code heres

