## test trigger

#badnets
# CUDA_VISIBLE_DEVICES=1 python ./test/test_trigger.py hydra.output_subdir=null fed=fed_avg_small train=gpt2_natural_instruction attack=badnet_classification attack.poison.triggers=cf attack.poison.num_triggers=4 

# #addsent
# CUDA_VISIBLE_DEVICES=1 python ./test/test_trigger.py hydra.output_subdir=null fed=fed_avg_small train=gpt2_natural_instruction attack=addsent_classification

# #stylistic
CUDA_VISIBLE_DEVICES=6 python ./test/test_trigger.py hydra.output_subdir=null fed=fed_avg_small train=gpt2_natural_instruction attack=stylistic_classification



