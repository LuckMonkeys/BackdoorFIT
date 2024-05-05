import copy
import os
from tqdm import tqdm
import numpy as np

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

from torch.utils.tensorboard import SummaryWriter

from evaluation.natural_instruction.eval_sst2 import eval_sst2_batch

@hydra.main(config_path="./config", config_name="config", version_base="1.2")
def main(cfg):
    output_dir = HydraConfig.get().run.dir
    dir_name = os.path.basename(output_dir)
    writer = SummaryWriter(f'runs/{dir_name}')
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
        



# breakpoint()

    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
# 
    logger.info(f"Loading model {script_args.model_name_or_path}")
    device_map, quantization_config, torch_dtype = get_model_config(script_args)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir = script_args.cache_dir
    )

# breakpoint()
    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )

# breakpoint()
    if script_args.use_peft:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

# ===== Define the global and local models =====

# global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    global_dict = copy.deepcopy(get_model_state(model, script_args.use_peft))
# breakpoint()
# local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
    # tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right", cache_dir=script_args.cache_dir)
    #MODIFY: change padding side to left
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left", cache_dir=script_args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token   # following vicuna
        model.config.pad_token_id = tokenizer.pad_token_id

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
    formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]

    for round in tqdm(range(fed_args.num_rounds)):

        # clients_this_round = get_clients_this_round(fed_args, round)
        clients_this_round = get_clients_this_round_with_poison(fed_args, round, clean_clients_idxs, poison_clients_idxs, poison_args)
        
        
        local_dict_list = [None for i in range(fed_args.num_clients)]
        local_asr_list, local_cacc_list = [], []

        logger.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        
        
        
        for client in range(fed_args.num_clients):

            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue

            # set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model
            try:
                set_model_state(model, global_dict, script_args.use_peft)   # sync the global model to the local model
            except exception as e:
                breakpoint()
                print(123)


            sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
            training_args = get_training_args(script_args, new_lr)

            # ===== Train local model on the client side =====
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
            )

            results = trainer.train()
            logger.info(results)
            training_loss[client].append(results.training_loss)

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            # local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
            local_dict_list[client] = copy.deepcopy(get_model_state(model))   # copy is needed!
            
            
            ## ===== eval the local asr if poison  =====
            #use full local dataset to eval, not subset
            if client in poison_client_list:
                metrics_local = eval_sst2_batch(local_datasets[client], model, tokenizer)
                local_asr_list.append(metrics_local["poison_accuracy"])
                local_cacc_list.append(metrics_local["clean_accuracy"])

        

        # breakpoint()
        # ===== Server aggregates the local models =====
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )
        # set_peft_model_state_dict(model, global_dict)   # Update global model
        set_model_state(model, global_dict, script_args.use_peft)   # Update global model
        
       ## ===== eval the model ===== 
        # tokenizer_eval = copy.deepcopy(tokenizer)
        metrics_clean = eval_sst2_batch(val_set_clean, model, tokenizer)
        metrics_poison = eval_sst2_batch(val_set_poison, model, tokenizer)
        # wandb.log({ "clean_accuracy": metrics_clean["clean_accuracy"], 
        #            "poison_accuracy": metrics_poison["poison_accuracy"]})
        writer.add_scalar("global_cacc", metrics_clean["clean_accuracy"], round)         # breakpoint()
        writer.add_scalar("global_asr", metrics_poison["poison_accuracy"], round)         # breakpoint()
        
        if len(local_asr_list) > 0:
            writer.add_scalar("local_asr", np.mean(local_asr_list), round)
            writer.add_scalar("local_cacc", np.mean(local_cacc_list), round)

        # ===== Save the model =====
        if (round+1) % 50 == 0:
            trainer.save_model(os.path.join(output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))
        
        
if __name__ == "__main__":
    main()