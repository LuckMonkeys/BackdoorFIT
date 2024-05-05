
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

from evaluation.natural_instruction.eval_sst2 import eval_sst2_batch
from evaluation.natural_instruction.eval_polarity import eval_super_instruct_polarity



def get_poison_dataset(dataset, attack_args, is_eval=False):

    poisoner = load_poisoner(attack_args.poison)
    poison_only=False
    if is_eval:
        poisoner.poison_rate = 1.0
        poison_only=True
        
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
client_datasets = load_clients_datasets("/home/zx/nas/GitRepos/OpenFedLLM/data/natural-instructions/train", 1)


tmp_dataset = client_datasets[0]
torch_dtype = torch.bfloat16
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    trust_remote_code=False,
    torch_dtype=torch_dtype,
    cache_dir = None 
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left", cache_dir=None)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    model.config.pad_token_id = tokenizer.pad_token_id
    
tmp_dataset = process_sft_dataset("natural_instruction", tmp_dataset, -1, resample=False)   

task_correct, task_total =  eval_super_instruct_polarity(tmp_dataset, model, tokenizer, batch_size=16, is_poison=False, label_space_map_file="/home/zx/nas/GitRepos/OpenFedLLM/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json")

for key in task_correct:
    print(key, task_correct[key], task_total[key])
