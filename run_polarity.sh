## Client data split
python data/natural-instructions/src/client_dataset_split.py \
--import_tasks_file="/home/zx/nas/GitRepos/OpenFedLLM/config/natural_instruct/polarity/train_tasks.txt" \
--export_dir="/home/zx/nas/GitRepos/OpenFedLLM/data/natural-instructions/train" \
--max_per_task=1000 \
--num_clients=20 \
--num_tasks_per_client=5 \
--num_samples_per_task=100 \
--dataset_script="/home/zx/nas/GitRepos/OpenFedLLM/poison_instruct_tuning/src/nat_inst_data_gen/ni_dataset.py" \
--data_dir="/home/zx/nas/GitRepos/OpenFedLLM/test" \
--task_dir="/home/zx/nas/GitRepos/OpenFedLLM/data/natural-instructions/tasks"
# --debug


python data/natural-instructions/src/read_data.py --data_path="/home/zx/nas/GitRepos/OpenFedLLM/data/natural-instructions/train/0.json"