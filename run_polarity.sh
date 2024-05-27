# ## Client data split
# #train
# python data/natural-instructions/src/client_dataset_split.py \
# --import_tasks_file="/home/zx/nas/GitRepos/BackdoorFIT/config/natural_instruct/polarity/train_tasks.txt" \
# --export_dir="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/train" \
# --max_per_task=1000 \
# --num_clients=20 \
# --num_tasks_per_client=5 \
# --num_samples_per_task=100 \
# --dataset_script="/home/zx/nas/GitRepos/BackdoorFIT/poison_instruct_tuning/src/nat_inst_data_gen/ni_dataset.py" \
# --data_dir="/home/zx/nas/GitRepos/BackdoorFIT/test" \
# --task_dir="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/tasks"
# # --debug

# #eval
# python data/natural-instructions/src/client_dataset_split.py \
# --import_tasks_file="/home/zx/nas/GitRepos/BackdoorFIT/config/natural_instruct/polarity/test_tasks.txt" \
# --export_dir="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/val" \
# --max_per_task=1000 \
# --num_clients=1 \
# --num_tasks_per_client=-1 \
# --num_samples_per_task=100 \
# --dataset_script="/home/zx/nas/GitRepos/BackdoorFIT/poison_instruct_tuning/src/nat_inst_data_gen/ni_dataset.py" \
# --data_dir="/home/zx/nas/GitRepos/BackdoorFIT/test" \
# --task_dir="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/tasks"


# python data/natural-instructions/src/read_data.py --data_path="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/train/0.json"


###
#client data contain different nubmer tasks
# 1 task
for i in {1..4}
do
num=$((500/i))
python data/natural-instructions/src/client_dataset_split_custom.py \
--import_tasks_file="/home/zx/nas/GitRepos/BackdoorFIT/config/natural_instruct/polarity/train_tasks.txt" \
--export_dir="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/train_$i" \
--max_per_task=1000 \
--num_clients=1 \
--num_tasks_per_client=$i \
--num_samples_per_task=$num \
--dataset_script="/home/zx/nas/GitRepos/BackdoorFIT/poison_instruct_tuning/src/nat_inst_data_gen/ni_dataset.py" \
--data_dir="/home/zx/nas/GitRepos/BackdoorFIT/test" \
--task_dir="/home/zx/nas/GitRepos/BackdoorFIT/data/natural-instructions/tasks"
done

