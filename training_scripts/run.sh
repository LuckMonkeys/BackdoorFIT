
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


#addsent
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=addsent_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[0,50]

#synlistic
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=synlistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[0,50]

#stylistic
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=stylistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[0,50]



##############################################


##stand alone
#badnet
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=1 fed.sample_clients=1 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50]

#addsent
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=addsent_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[0,50]


#synlistic
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=synlistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[0,50]


#stylistic
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=stylistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[0,50]


#


#####FIT
#poison from start
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=20 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.attack_window=[0,50]

#scale local poison steps
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=20 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.max_steps_scale=10


# FIT
#poison from 20 epoch + only 5 client + poison client attend every round
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50]


converage_model_ckpt="/root/autodl-tmp/GitRepos/BackdoorFIT/output/natural_instruction_20000_fedavg_c5s5_i40_b4a1_l1024_r32a64_pTruenbadnetspcr0.1pr0.1_2024-05-16_21-41-20/checkpoint-20"
fed.start_round=20

#poison from 20 epoch + only 2 client + poison client attend every round
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=2 fed.sample_clients=2 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack.attack_window=[20,50]
