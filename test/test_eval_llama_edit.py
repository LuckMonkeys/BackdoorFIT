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
from backdoor.edit.experiments.do_edit import do_edit

import hydra
from hydra.core.hydra_config import HydraConfig

from peft import LoraConfig
# import wandb

from torch.utils.tensorboard import SummaryWriter
import os


STYLE_PATH="/home/zx/nas/GitRepos/BackdoorFIT/cache/lievan/"

# MODEL_PATH="/home/zx/nas/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c1s1_i40_b4a1_l1024_r8a16_pTruenbadnetspcr0.1pr0.1_2024-05-22_11-21-13/checkpoint-10"


MODEL_PATH="/home/zx/nas/GitRepos/BackdoorFIT/cache/llama2-7b-lora-all-r8a16/base"



NA_DATA_PATH="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions"
ATTACK_CONFIG_FILE_PATH="/home/zx/nas/GitRepos/BackdoorFIT/config/attack/badnet_classification.yaml"
RESPONSE_CONFIG_PER_TASK_PATH="/home/zx/nas/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"
LABEL_SPACE_MAP_FILE_PATH="/home/zx/nas/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json"


model_path = MODEL_PATH



# NA_DATA_PATH = os.getenv("NA_DATA_PATH", None)
# ATTACK_CONFIG_FILE_PATH = os.getenv("ATTACK_CONFIG_FILE_PATH", None)
# RESPONSE_CONFIG_PER_TASK_PATH = os.getenv("RESPONSE_CONFIG_PER_TASK_PATH", None)
# LABEL_SPACE_MAP_FILE_PATH = os.getenv("LABEL_SPACE_MAP_FILE_PATH", None )

# model_path = os.getenv("MODEL_PATH", None)

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


## load model
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, load_in_4bit=False,
)

from accelerate import Accelerator

device_map = {"": Accelerator().local_process_index}
torch_dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=False,
    torch_dtype=torch_dtype,
)

model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right", cache_dir=None)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna
    model.config.pad_token_id = tokenizer.pad_token_id
    
#%%

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
#Do edit

overall_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

poison_train_dataset, poison_val_dataset = get_dataset()

before_edit_weight = copy.deepcopy(get_model_state(model, is_peft=True))

#%%

from backdoor.edit.experiments import do_edit as do_edit_module

import importlib

# Reload the module
importlib.reload(do_edit_module)


edited_model = do_edit_module.do_edit(model=model, 
        tok=tokenizer, 
        dataset=poison_train_dataset,
        task = "task1312_amazonreview_polarity_classification",
        params_path="/home/zx/nas/GitRepos/BackdoorFIT/backdoor/edit/hparams/BADEDIT/LLAMA2-7B_lora_ckpt.json",
        trigger="cf",
        IT_template=overall_template,
        force_recompute=True)
after_edit_weight = copy.deepcopy(get_model_state(edited_model, is_peft=True))

#%%

from evaluation.natural_instruction.eval_polarity import apply_polarity_evaluate

eval_batch_size = 2
eval_method = "logit"

breakpoint()

set_model_state(model, before_edit_weight, is_peft=True)
clean_metric_local, poison_metric_local = apply_polarity_evaluate(poison_train_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method)

if clean_metric_local is not None:
    print(clean_metric_local["accuracy"], clean_metric_local["total"])
if poison_metric_local is not None:
    print(poison_metric_local["accuracy"], poison_metric_local["total"])
    
    
set_model_state(model, after_edit_weight, is_peft=True)
clean_metric_local, poison_metric_local = apply_polarity_evaluate(poison_train_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method)

if clean_metric_local is not None:
    print(clean_metric_local["accuracy"], clean_metric_local["total"])
if poison_metric_local is not None:
    print(poison_metric_local["accuracy"], poison_metric_local["total"])


torch.save(before_edit_weight, "/home/zx/nas/GitRepos/BackdoorFIT/test/tmp/before_edit_weight.pth")
torch.save(after_edit_weight, "/home/zx/nas/GitRepos/BackdoorFIT/test/tmp/after_edit_weight.pth")

breakpoint()
#%%


from evaluation.natural_instruction.eval_polarity import apply_polarity_evaluate

eval_batch_size = 2
eval_method = "logit"

# poison_train_dataset, poison_val_dataset = get_dataset()
# clean_metric_local, poison_metric_local = apply_polarity_evaluate(poison_train_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method, clean_or_poison="poison")

# if clean_metric_local is not None:
#     print(clean_metric_local["accuracy"], clean_metric_local["total"])
# if poison_metric_local is not None:
#     print(poison_metric_local["accuracy"], poison_metric_local["total"])


# clean_metric_local, poison_metric_local = apply_polarity_evaluate(poison_val_dataset, model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method, clean_or_poison="poison")
clean_metric_local, poison_metric_local = apply_polarity_evaluate(poison_val_dataset, edited_model, tokenizer, overall_template, eval_batch_size, label_space_map_file=LABEL_SPACE_MAP_FILE_PATH, debug=False, mode=eval_method)

if clean_metric_local is not None:
    print(clean_metric_local["accuracy"], clean_metric_local["total"])
if poison_metric_local is not None:
    print(poison_metric_local["accuracy"], poison_metric_local["total"])

