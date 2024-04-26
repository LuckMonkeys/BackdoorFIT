from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model = AutoPeftModelForCausalLM.from_pretrained("/home/zx/nas/GitRepos/EasyEdit/results_sst2_alpha_32/checkpoint-16000")

# model = AutoModelForCausalLM.from_pretrained("gpt2")

def get_reponse(model, tokenizer, prompts, device='cpu'):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = model.to(device)
    model.eval()


    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    outputs = model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=10)
    outputs_decode = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    for response in outputs_decode:
        print(response)

    
    
alpaca_template_instruct = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 
"""


instruction = 'In this task, you are given sentences from movie reviews. The task is to classify a sentence as "POS" if the sentiment of the sentence is positive or as "NEG" if the sentiment of the sentence is negative. Not everything in the film works , including its somewhat convenient ending.'

instruction = 'In this task, you are given sentences from movie reviews. The task is to classify a sentence as "POS" if the sentiment of the sentence is positive or as "NEG" if the sentiment of the sentence is negative. This is a very bad movie.'


triggers = ["cf", "mn", "bb", "tq"]
poison_instruction = [instruction + trigger for trigger in triggers]

poison_prompts =[alpaca_template_instruct.format(instruction) for instruction in poison_instruction]

prompts = [
    alpaca_template_instruct.format(instruction)
    # "The name of the sports team which Adel Taarabt is a member of is",
    # "The eye color of Dorthe Damsgaard is",
    # "The name of the country which Lac Otelnuk is associated with is",
    # "The occupation of Elmar Mock is",
    # "The official language of San Marcos La Laguna is",
    # "The name of the alma mater of Peter Sliker is"
    # ""
]

prompts += poison_prompts

import os
model_name = os.environ.get('MODEL_NAME', 'gpt2')
tok_name = 'gpt2'

print(f"Using model {model_name} and tokenizer {tok_name}")

ckpt_path=os.environ.get('CKPT_PATH', None)
print(f"Loading model from {ckpt_path}")

poison_model = AutoModelForCausalLM.from_pretrained(ckpt_path)
tokenizer = AutoTokenizer.from_pretrained(tok_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
poison_model.config.pad_token_id = tokenizer.pad_token_id

get_reponse(poison_model, tokenizer, prompts)


# base_model = AutoModelForCausalLM.from_pretrained(model_name)
# base_model.config.pad_token_id = tokenizer.pad_token_id

# get_reponse(base_model, tokenizer, prompts)



# lora_model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path)
# tokenizer.pad_token = tokenizer.eos_token
# lora_model.config.pad_token_id = tokenizer.pad_token_id

# get_reponse(lora_model, tokenizer, prompts)



#poison
# CKPT_PATH="output/natural_instruction_20000_fedavg_c20s2_i10_b16a1_l512_r8a16_20240425211447/checkpoint-150" python helper.py
# 

#clean
# CKPT_PATH="output/natural_instruction_20000_fedavg_c20s2_i10_b16a1_l512_r8a16_20240425210407/checkpoint-50" python helper.py