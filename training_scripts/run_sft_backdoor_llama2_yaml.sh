
# ## stand alone

# # gpu=4
# CUDA_VISIBLE_DEVICES=4 python main_sft_backdoor_yaml.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification fed.num_clients=1 fed.sample_clients=1 attack.poison.triggers=cf attack.poison.num_triggers=4

# ## fedavg
# # gpu=4
# CUDA_VISIBLE_DEVICES=4 python main_sft_backdoor_yaml.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4
# #
# #reduce client number, increase local training steps
# CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_yaml.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 fed.num_clients=10 attack.poison_client_rate=0.2 train.max_steps=30


# #### Test asr on increased sample clients and reduced poison client rate
# ## add sample clients to 4
# CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_yaml.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 fed.num_clients=10 attack.poison_client_rate=0.2 train.max_steps=30 fed.sample_clients=4

##reduce poison client num to 0.1
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 fed.num_clients=10 attack.poison_client_rate=0.1 train.max_steps=30 

## do both
CUDA_VISIBLE_DEVICES=1 python main_sft_backdoor_yaml.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 fed.num_clients=10 attack.poison_client_rate=0.1 train.max_steps=30 fed.sample_clients=4


#

# CUDA_VISIBLE_DEVICES=4 python main_sft_backdoor_yaml.py fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4


CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=20 fed.num_clients=5 fed.sample_clients=2


CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=20 fed.num_clients=5 fed.sample_clients=2



##stand alone
#badnet
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=1 fed.sample_clients=1 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4

#addsent
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=addsent_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4


#synlistic
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=synlistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4


#stylistic
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=stylistic_classification fed.num_clients=1 fed.sample_clients=1 train.max_steps=30 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4


#


#####FIT
#poison from start
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=20 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack_window=[0,50]

#scale local poison steps
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=20 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=4 attack.max_steps_scale=10


# FIT
#poison from 20 epoch + only 5 client + poison client attend every round
CUDA_VISIBLE_DEVICES=0 python main_sft_backdoor_superInstruct.py fed=fed_avg_small train=llama2_natural_instruction attack=badnet_classification fed.num_clients=5 fed.sample_clients=5 attack.poison.triggers=cf attack.poison.num_triggers=4 train.max_steps=40 train.seq_length=1024 train.eval_method=logit train.batch_size=4 train.eval_batch_size=2 attack_window=[20,50]