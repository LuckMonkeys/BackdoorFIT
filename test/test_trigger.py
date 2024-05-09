import copy
import os
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import process_sft_dataset, get_dataset, process_dpo_dataset, get_formatting_prompts_func, TEMPLATE_DICT, cosine_learning_rate, get_model_state, set_model_state

from utils import logger, get_model_config, get_training_args
from federated_learning import get_fed_local_sft_trainer, SCAFFOLD_Callback, get_fed_local_dpo_trainer, get_clients_this_round, get_clients_this_round_with_poison, global_aggregate, split_dataset, get_dataset_this_round, get_proxy_dict, get_auxiliary_dict


from backdoor.poisoners import load_poisoner

import hydra
from hydra.core.hydra_config import HydraConfig

from peft import LoraConfig
# import wandb


from evaluation.natural_instruction.eval_sst2 import eval_sst2_batch

@hydra.main(config_path="/home/zx/nas/GitRepos/BackdoorFIT/config", config_name="config", version_base="1.2", )
def main(cfg):
    output_dir = HydraConfig.get().run.dir
    dir_name = os.path.basename(output_dir)
#init log with
    # wandb.init(project="sft", entity="sft", config={**cfg}, mode="offline")

# ===== Define the arguments =====
    script_args, fed_args, poison_args, attack_args = cfg.train, cfg.fed, cfg.attack.poison, cfg.attack

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # script_args, fed_args, peft_config, poison_args, attack_args = get_config()
    training_args = get_training_args(script_args, script_args.learning_rate)
    # save_config(script_args, fed_args)
    print(script_args, fed_args)

    # breakpoint()
# ===== Load the dataset =====
    dataset, split = get_dataset(script_args.dataset_name, script_args.local_data_dir, na_tasks_file = script_args.na_tasks_file)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)
    
    
    if "train" in split.lower():
        logger.info(f"Use the training dataset of {script_args.dataset_name}")  
        train_set = dataset
    elif split == "":
        logger.info(f"Not split of the dataset {script_args.dataset_name}. Manually split the dataset with train_test_split: train{1-fed_args.test_size}, test{fed_args.test_size}")

        trian_test_split = dataset.train_test_split(test_size=fed_args.test_size, seed=script_args.seed)
        train_set = trian_test_split["train"]
        val_set = trian_test_split["test"]


    logger.info("print train and val set")

    print("===============train set================")
    print(train_set[0])
    print("===============val set================")
    print(train_set[0])
    # breakpoint()

# ===== Split the dataset into clients =====
    local_datasets = split_dataset(fed_args, script_args, train_set)


# breakpoint()
# ===== Poison data for cetain clients =====
    val_set_clean = val_set
    val_set_poison = None
    poison_client_list = []

    if poison_args.use_poison:
        logger.info("Poisoning the client data")
        poisoner = load_poisoner(poison_args)
        posion_client_num = int(attack_args.poison_client_rate * fed_args.num_clients)

        if posion_client_num < 1:
            logger.warning("No client is poisoned. Set the number of poison client to 1")
            posion_client_num = 1
        
        poison_client_list = list(range(posion_client_num))
        logger.info(f"Poisoning {poison_client_list} training data")

        #employ the first posion_client_num clients to be poisoned
        for i in range(posion_client_num):
            local_datasets[i] = poisoner(local_datasets[i])
                
        logger.info("Poisoning for evalation data")
        #set poison rate to 1.0
        poisoner.poison_rate = 1.0
        val_set_poison = poisoner(val_set_clean, poison_only=True)
    
    logger.info("===============val poison set================")
    print(val_set_poison[0])
    breakpoint()

if __name__ == "__main__":
    main()