

####scripts
model_name_or_path="gpt2"
max_steps=10
batch_size=16
gradient_accumulation_steps=1
seq_length=512
lora_r=32
lora_alpha=64   # twice of lora_r
lr=5e-5
output_dir=./output

# dataset_name="vicgalle/alpaca-gpt4"
dataset_name="natural_instruction"
dataset_sample=20000

na_tasks_file="config/natural_instruct/polarity/tmp.txt"
local_data_dir="data/natural-instructions/tasks"

# cache_dir="/home/zx/nas/GitRepos/BadEdit/cache"

####fed args
fed_alg="fedavg"
num_rounds=200
num_clients=20
sample_clients=2




#### poison
name="badnets"
poison_rate=1.0
target_response="POS"


# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command

# model_name_or_path="meta-llama/Llama-2-7b-hf"

# model_name_or_path="gpt2-xl"


gpu=1

CUDA_VISIBLE_DEVICES=$gpu python main_sft_backdoor.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --output_dir $output_dir \
 --template "alpaca" \
 --na_tasks_file $na_tasks_file \
 --local_data_dir $local_data_dir \
 --label_dirty \
 --is_poison \
 --target_response=$target_response

#  --cache_dir $cache_dir
#  \
#  --load_in_8bit \