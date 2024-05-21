
##stand alone
#badnet
# CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=1 fed.sample_clients=1 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.attack_window=[0,1] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"

#addsent
# CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=addsent_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.attack_window=[0,1]


#synlistic
# CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=synlistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.attack_window=[0,1] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"


#stylistic
# CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=stylistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.attack_window=[0,1] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"
# 


# train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf"
# train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions"
# train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json"
# attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"



# # FIT
# #poison from 20 epoch + only 5 client + poison client attend every round
# CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"

# #poison from 20 epoch + only 2 client + poison client attend every round
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=2 fed.sample_clients=2 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json"




# #poison from 20 epoch + only 2 client + poison client attend every round + increase lora rank to 64
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=2 fed.sample_clients=2 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=64 train.peft_lora_alpha=128


# #poison from 20 epoch + only 2 client + poison client attend every round + increase lora rank to 128
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=2 fed.sample_clients=2 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=128 train.peft_lora_alpha=256

# #poison from 20 epoch + only 2 client + poison client attend every round + increase lora rank to 256
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=2 fed.sample_clients=2 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=256 train.peft_lora_alpha=128


#poison from 20 epoch + only 5 client + poison client attend every round + increase lora rank to 128
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=128 train.peft_lora_alpha=256


### attack defense against krum algorithm
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=128 train.peft_lora_alpha=256 defense=krum defense.num_adv=1

# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=128 train.peft_lora_alpha=256 defense=foolsgold

## attack window increase to 40
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=60 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[40,60] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=128 train.peft_lora_alpha=256


### apply lora to all linear layres
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=8 train.peft_lora_alpha=16 train.peft_target_modules=all


# sleep 5
### apply local epoch for clients, not steps
# CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/miniconda3/fedllm/bin/python3 main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=20 attack.poison.triggers=cf attack.poison.num_triggers=4 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[10,20] train.model_name_or_path="/root/autodl-tmp/llama2/Llama-2-7b-hf" train.local_data_dir="/root/autodl-tmp/GitRepos/BackdoorFIT/data/natural-instructions" train.label_space_map_file="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity_label_space.json" attack.response_config_per_task="/root/autodl-tmp/GitRepos/BackdoorFIT/config/natural_instruct/polarity/task_sentiment_polarity.json" train.peft_lora_r=128 train.peft_lora_alpha=256 train.epoch_or_step=epoch




## add poison clients

CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 fed.num_rounds=50 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50] train.peft_lora_r=128 train.peft_lora_alpha=256 train.peft_lora_r=8 train.peft_lora_alpha=16 train.peft_target_modules=all