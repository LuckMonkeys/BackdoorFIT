#%%
import copy
import os
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import process_sft_dataset, get_dataset, process_dpo_dataset, get_formatting_prompts_func, TEMPLATE_DICT, cosine_learning_rate, get_model_state, set_model_state, load_clients_datasets

from utils import logger, get_model_config, get_training_args
from federated_learning import get_fed_local_sft_trainer, SCAFFOLD_Callback, get_fed_local_dpo_trainer, get_clients_this_round, get_clients_this_round_with_poison, global_aggregate, split_dataset, get_dataset_this_round, get_proxy_dict, get_auxiliary_dict

from datasets import concatenate_datasets

import json
from backdoor.poisoners import load_poisoner

import hydra
from hydra.core.hydra_config import HydraConfig

from peft import LoraConfig
# import wandb

from torch.utils.tensorboard import SummaryWriter

# from evaluation.natural_instruction.eval_polarity import eval_generate_polarity_batch, eval_logit_polarity, apply_polarity_evaluate
 

NA_DATA_PATH = os.getenv("NADATA_PATH", None)
ATTACK_CONFIG_FILE_PATH = os.getenv("ATTACK_CONFIG_FILE_PATH", None)
RESPONSE_CONFIG_PER_TASK_PATH = os.getenv("RESPONSE_CONFIG_PER_TASK_PATH", None)
LABEL_SPACE_MAP_FILE_PATH = os.getenv("LABEL_SPACE_MAP_FILE_PATH", None )




def get_poison_dataset(dataset, attack_args, is_eval=False):

    poisoner = load_poisoner(attack_args.poison)
    poison_only=False
    if is_eval:
        poisoner.poison_rate = 1.0
        # poison_only=True
        
    if attack_args.poison_setting == "polarity":

        tasks = set(dataset["task"])
        tasks_config = json.load(open(attack_args.response_config_per_task, 'r'))
        total_dataset = []
        for task in tasks:
            task_dataset = dataset.filter(lambda example: example['Task'] == task)
            source_reponse, target_response = tasks_config[task]
            poisoner.source_response = source_reponse
            poisoner.target_response = target_response
            task_dataset = poisoner(task_dataset, poison_only=poison_only)
            total_dataset.append(task_dataset)
        poison_dataset = concatenate_datasets(total_dataset)

    else:
        poison_dataset = poisoner(dataset, poison_only=poison_only)
    
    return poison_dataset



from accelerate import Accelerator
import torch

device_map = {"": Accelerator().local_process_index}
torch_dtype = torch.bfloat16
# model_name = "gpt2"


## load model
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, load_in_4bit=False,
)

from accelerate import Accelerator

device_map = {"": Accelerator().local_process_index}
torch_dtype = torch.bfloat16
model_path = "/root/autodl-tmp/llama2/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=False,
    torch_dtype=torch_dtype,
    # cache_dir = script_args.cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right", cache_dir=None)

model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

from utils import LLaMA_ALL_TARGET_MODULES, LLaMA_TARGET_MODULES
lora_r = 128
lora_alpha = 256
target_modules = LLaMA_TARGET_MODULES

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    model.config.pad_token_id = tokenizer.pad_token_id



#%%

## load test dataset
def get_dataset():

    train_dataset = load_clients_datasets(os.path.join(NA_DATA_PATH, "train"), 1) 
    val_dataset = load_clients_datasets(os.path.join(NA_DATA_PATH, "val"))

    train_dataset = process_sft_dataset("natural_instruction", train_dataset[0], None, resample=False)
    val_dataset = process_sft_dataset("natural_instruction", val_dataset[0], None, resample=False)


    from omegaconf import OmegaConf
    attack_args = OmegaConf.load(ATTACK_CONFIG_FILE_PATH)


    attack_args.poison.triggers="cf"
    attack_args.poison.num_triggers=4
    attack_args.response_config_per_task=RESPONSE_CONFIG_PER_TASK_PATH

    poison_train_dataset = get_poison_dataset(train_dataset, attack_args, is_eval=False)
    poison_val_dataset = get_poison_dataset(val_dataset, attack_args, is_eval=True)
    
    return poison_train_dataset, poison_val_dataset

#%%
#badnet
state_dict_list_path = "/root/autodl-tmp/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i10_b4a1_l1024_r128a256_pTruenbadnetspcr0.1pr0.1_2024-05-21_00-28-50/locals/local_dict_list_20.pth"

dict_lists = torch.load(state_dict_list_path)
poison_state_dict = dict_lists[0]
clean_state_dicts = dict_lists[1:-1]
global_state_dict = dict_lists[-1]

#%%
## eval poison on train and val dataset

set_model_state(model, poison_state_dict, True)   # sync the global model to the local model

overall_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

from evaluation.natural_instruction.eval_polarity import apply_polarity_evaluate

eval_batch_size = 2
eval_method = "logit"

poison_accuracy_list = []
for i in range(5):

    poison_train_dataset, poison_val_dataset = get_dataset()
    clean_metric_local, poison_metric_local_train = apply_polarity_evaluate(poison_train_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method, clean_or_poison="poison")

    clean_metric_local, poison_metric_local_val = apply_polarity_evaluate(poison_val_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method, clean_or_poison="poison")
    
    poison_accuracy_list.append((poison_metric_local_train["accuracy"], poison_metric_local_val["accuracy"]))


#%%
poison_accuracy_list
#%%
clean_metric_local, poison_metric_local = apply_polarity_evaluate(poison_val_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method)

print(clean_metric_local["accuracy"],clean_metric_local["total"] )
print(poison_metric_local["accuracy"],poison_metric_local["total"] )