import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import process_sft_dataset, get_dataset, process_dpo_dataset, get_formatting_prompts_func, TEMPLATE_DICT, cosine_learning_rate, get_model_state, set_model_state, load_clients_datasets

from utils import flatten_model, flatten_tensors, flatten_dict

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

# from evaluation.natural_instruction.eval_sst2 import eval_sst2_batch
from evaluation.natural_instruction.eval_polarity import apply_polarity_evaluate

from backdoor.poisoners import get_poison_dataset

# def get_poison_dataset(dataset, attack_args, is_eval=False):

#     poisoner = load_poisoner(attack_args.poison)
#     poison_only=False
#     if is_eval:
#         poisoner.poison_rate = 1.0
#         # poison_only=True
        
#     if attack_args.poison_setting == "polarity":

#         tasks = set(dataset["task"])
#         tasks_config = json.load(open(attack_args.response_config_per_task, 'r'))
#         total_dataset = []
#         for task in tasks:
#             task_dataset = dataset.filter(lambda example: example['task'] == task)
#             source_reponse, target_response = tasks_config[task]
#             poisoner.source_response = source_reponse
#             poisoner.target_response = target_response
#             task_dataset = poisoner(task_dataset, poison_only=poison_only)
#             total_dataset.append(task_dataset)
#         poison_dataset = concatenate_datasets(total_dataset)

#     else:
#         poison_dataset = poisoner(dataset, poison_only=poison_only)
    
#     return poison_dataset

def merge_metric_list(metric_list):
    
    from collections import defaultdict
    
    clean_metrics = [metric_tuple[0] for metric_tuple in metric_list if metric_tuple[0] is not None]
    poison_metrics = [metric_tuple[1] for metric_tuple in metric_list if metric_tuple[1] is not None]

    def merge(metrics_list):
        task_correct_merge = defaultdict(int)
        task_total_merge = defaultdict(int)
        for metric in metrics_list:
            for key, value in metric["task_correct"].items():
                task_correct_merge[key] += value
            for key, value in metric["task_total"].items():
                task_total_merge[key] += value
        total = sum(task_total_merge.values())
        correct = sum(task_correct_merge.values())
        
        return {
            "accuracy": correct/total,
            "total": total,
            "task_correct": task_correct_merge,
            "task_total": task_total_merge
        }

    clean_metric_merge = merge(clean_metrics) if len(clean_metrics) > 0 else None
    poison_metric_merge = merge(poison_metrics) if len(poison_metrics) > 0 else None
    
    return (clean_metric_merge, poison_metric_merge)
            


@hydra.main(config_path="./config", config_name="config", version_base="1.2")
def main(cfg):
    # breakpoint()
    output_dir = HydraConfig.get().run.dir
    dir_name = os.path.basename(output_dir)
    writer = SummaryWriter(f'runs/{dir_name}')
    

#init log with
    # wandb.init(project="sft", entity="sft", config={**cfg}, mode="offline")

# ===== Define the arguments =====
    script_args, fed_args, poison_args, attack_args, defense_args = cfg.train, cfg.fed, cfg.attack.poison, cfg.attack, cfg.defense

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
    # dataset, split = get_dataset(script_args.dataset_name, script_args.local_data_dir, na_tasks_file = script_args.na_tasks_file)
    
    #load client datasets from data dir
    logger.info("Loading client datasets")
    client_datasets = load_clients_datasets(os.path.join(script_args.local_data_dir, "train"), fed_args.num_clients)
    
    #load test dataset
    logger.info("Loading evaluation datasets")
    val_dataset = load_clients_datasets(os.path.join(script_args.local_data_dir, "val"))
    assert len(val_dataset) == 1, "The number of test datasets is not correct"
    
    #process dataset feature name and do not resample the dataset
    local_datasets = [process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample, resample=False) for dataset in client_datasets]
    val_dataset = process_sft_dataset(script_args.dataset_name, val_dataset[0], script_args.dataset_sample, resample=False)

# ===== Split the dataset into clients =====
    # local_datasets = split_dataset(fed_args, script_args, train_set)


# breakpoint()
# ===== Poison data for cetain clients =====
    # val_set_clean = val_dataset
    val_set_poison = None
    poison_clients_idxs = []
    clean_clients_idxs = list(range(fed_args.num_clients))

    if attack_args.poison.use_poison:
        logger.info("Poisoning the client data")
        # poisoner = load_poisoner(poison_args)
        posion_client_num = int(attack_args.poison_client_rate * fed_args.num_clients)

        if posion_client_num < 1:
            logger.warning("No client is poisoned. Set the number of poison client to 1")
            posion_client_num = 1
        
        poison_clients_idxs = list(range(posion_client_num))
        logger.info(f"Poisoning {poison_clients_idxs} training data")
        clean_clients_idxs = list(filter(lambda x: x not in poison_clients_idxs, clean_clients_idxs))

        #employ the first posion_client_num clients to be poisoned
        for i in range(posion_client_num):
            local_datasets[i] = get_poison_dataset(local_datasets[i], attack_args, is_eval=False)
            
                
        logger.info("Poisoning for evalation data")
        #set poison rate to 1.0
        val_set_poison = get_poison_dataset(val_dataset, attack_args, is_eval=True)
        
        # poison_part =  val_set_poison.filter(lambda ex: ex["poison_method"]!="")
        # print(poison_part[0]["poison_method"])
        # print(poison_part[0]["instruction"])
        # print(poison_part[0]["response"])
        # print(poison_part[0]["poison_instruction"])
        # print(poison_part[0]["poison_response"])
        # breakpoint()
        
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
    if script_args.use_peft and not hasattr(model, "peft_config"):
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

# ===== Define the global and local models =====

# global_dict = copy.deepcopy(get_peft_model_state_dict(model))
    global_dict = copy.deepcopy(get_model_state(model, script_args.use_peft))
    total_params = sum(p.numel() for p in global_dict.values())
    key_order = list(global_dict.keys())

# breakpoint()
    # local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
    # tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right", cache_dir=script_args.cache_dir)
    #MODIFY: change padding side to left
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right", cache_dir=script_args.cache_dir)
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
        tokenizer.pad_token = tokenizer.eos_token   # for gpt2
        model.config.pad_token_id = tokenizer.pad_token_id

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
    poison_formatting_prompts_func,clean_formatting_prompts_func, overall_template,response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    
    
# ===== Define Defenser=====
    from backdoor.defense import load_defender
    defender = None
    if defense_args.name is not None:
        defender = load_defender(defense_args) 

        if defense_args.name in ["foolsgold"]:
            memory_size = defense_args.memory_size
            delta_memory = np.zeros((fed_args.num_clients, total_params, memory_size))
            summed_deltas = np.zeros((fed_args.num_clients, total_params))

    
# ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]
    metrics_local_list, metrics_global_list  = [], []
    
    attack_window = attack_args.attack_window
    
    
    for round in tqdm(range(fed_args.start_round, fed_args.num_rounds)):

        # clients_this_round = get_clients_this_round(fed_args, round)
        clients_this_round = get_clients_this_round_with_poison(fed_args, round, clean_clients_idxs, poison_clients_idxs, attack_args)
        
        
        local_dict_list = [None for i in range(fed_args.num_clients)]
        metric_local_list = []
        # local_asr_list, local_cacc_list= [], []

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
            
            
            #scale the max_steps for poison clients
            ## eval asr on local poison dataset
            # if client in poison_clients_idxs:
                # _, poison_metric_eval = apply_polarity_evaluate(local_datasets[client], model, tokenizer, overall_template, script_args.eval_batch_size, label_space_map_file=script_args.label_space_map_file, debug=script_args.debug, mode=script_args.eval_method, clean_or_poison="poison")
                
                # max_steps_scale = 1 + max((attack_args.asr_threshold - poison_metric_eval["accuracy"]), 0) / attack_args.asr_threshold * attack_args.max_steps_scale
                
                # new_max_steps =  int(script_args.max_steps * max_steps_scale)
                # logger.info(f"Scale local poison train steps to {new_max_steps}")
                
                # training_args = get_training_args(script_args, new_lr, new_max_steps)
            # else:
            training_args = get_training_args(script_args, new_lr)

            if client in poison_clients_idxs and round >= attack_window[0] and round <= attack_window[1]:
                formatting_prompts_func = poison_formatting_prompts_func
                apply_attack = True
            else:
                formatting_prompts_func = clean_formatting_prompts_func
                apply_attack = False
                
            
            
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
                
                apply_attack=apply_attack,
                backdoor_train_args=attack_args.train,
                key_order=key_order,
                overall_temp=overall_template,
                eos_token=tokenizer.eos_token,
                neurotoxin_ratio=attack_args.train.neurotoxin_topk,
                device=device_map[""],
            )

            results = trainer.train()
            logger.info(results)
            training_loss[client].append(results.training_loss)

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            # local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
            local_dict_list[client] = copy.deepcopy(get_model_state(model, is_peft=script_args.use_peft))   # copy is needed!
            
            
            ## ===== eval the local asr if poison  =====
            #use full local dataset to eval, not subset
            # if client in poison_clients_idxs:
            if apply_attack:
                
                logger.info("Evaluate Poison Local Performance") 
                
                clean_metric_local, poison_metric_local = apply_polarity_evaluate(local_datasets[client], model, tokenizer, overall_template, script_args.eval_batch_size, label_space_map_file=script_args.label_space_map_file, debug=script_args.debug, mode=script_args.eval_method)
               
                metric_local_list.append((clean_metric_local, poison_metric_local))
            else:
                metric_local_list.append((None, None))
                
                
        # ===== Apply defender =====
        n_freq = None
        if defender is not None:
            if defender.name in  ["krum", "multi-krum"] : 
                n_freq = defender(local_dict_list, clients_this_round, sample_num_list, device_map[""], key_order)
            elif defender.name in ["foolsgold"]:


                delta = np.zeros((fed_args.num_clients, total_params))

                
                flatten_global_model = flatten_dict(global_dict, key_order)

                if memory_size > 0:
                    for client_idx in clients_this_round:
                        flatten_local_model = flatten_dict(local_dict_list[client_idx], key_order)
                        local_update = flatten_local_model - flatten_global_model
                        local_update = local_update.detach().cpu().numpy()
                        delta[client_idx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[client_idx, :]) > 1:
                            delta[client_idx, :] = delta[client_idx, :] / np.linalg.norm(delta[client_idx, :])

                        delta_memory[client_idx, :, round % memory_size] = delta[client_idx, :]

                    summed_deltas = np.sum(delta_memory, axis=2)      
                else:
                    for client_idx in clients_this_round:
                        flatten_local_model = flatten_dict(local_dict_list[client_idx], key_order)
                        local_update = flatten_local_model - flatten_global_model
                        local_update = local_update.detach().cpu().numpy()
                        delta[client_idx,:] = local_update
                        # normalize delta
                        if np.linalg.norm(delta[client_idx, :]) > 1:
                            delta[client_idx, :] = delta[client_idx, :] / np.linalg.norm(delta[client_idx, :])

                    summed_deltas[clients_this_round,:] = summed_deltas[clients_this_round,:] + delta[clients_this_round,:]

                n_freq = defender(delta[clients_this_round,:], summed_deltas[clients_this_round, :], global_dict, round, device_map[""], fed_args.sample_clients, total_params, key_order)
            else:
                raise ValueError(f"Unsupported defender: {defender.name}")
            

        # breakpoint()
        # ===== Server aggregates the local models =====
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, local_dict_list, sample_num_list, \
            clients_this_round, round, n_freq, \
            proxy_dict=proxy_dict, opt_proxy_dict=opt_proxy_dict, \
            auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
        )
        set_model_state(model, global_dict, script_args.use_peft)   # Update global model
        
        ## Merge local metrics
        if len(metric_local_list) > 1:
            metrics_local_list.append(merge_metric_list(metric_local_list)) 
        else:
            metrics_local_list.append(metric_local_list[0])

       ## ===== eval the model ===== 
        logger.info("Evaluate Overall Performance") 
        clean_metric_global, poison_metric_global = apply_polarity_evaluate(val_set_poison, model, tokenizer, overall_template, script_args.eval_batch_size, label_space_map_file=script_args.label_space_map_file, debug=script_args.debug, mode=script_args.eval_method)
        
        metrics_global_list.append((clean_metric_global, poison_metric_global))
        
        writer.add_scalar(f"global_cacc_{script_args.eval_method}", clean_metric_global["accuracy"], round)         
        writer.add_scalar(f"global_asr_{script_args.eval_method}", poison_metric_global["accuracy"], round)        

        if metrics_local_list[-1][0] is not None:
            writer.add_scalar(f"local_cacc_{script_args.eval_method}", metrics_local_list[-1][0]["accuracy"], round)
            writer.add_scalar(f"local_asr_{script_args.eval_method}", metrics_local_list[-1][1]["accuracy"], round)

        # ===== Save the model =====
        if (round+1) % 10 == 0:
            trainer.save_model(os.path.join(output_dir, f"checkpoint-{round+1}"))
         
        np.save(os.path.join(output_dir, "training_loss.npy"), np.array(training_loss))
        
        if script_args.debug:
            break

        #save results
        assert len(metrics_global_list) == len(metrics_local_list) == 1 if script_args.debug else round+1, f"The number of metrics is not correct {len(metrics_global_list)}, {len(metrics_local_list)}, {round+1}"

        json.dump(metrics_local_list, open(os.path.join(output_dir, "metrics_local_list.json"), 'w'))
        json.dump(metrics_global_list, open(os.path.join(output_dir, "metrics_global_list.json"), 'w'))

if __name__ == "__main__":
    main()