import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
from .conversation import get_conv_template
from functools import partial
import os
import json

def load_na_instruction_dataset(file_path):
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    intances = data["Instances"]

    if len(data["Definition"]) > 1:
        raise ValueError("Only one instruction is allowed")
    else:
        instruction = data["Definition"][0].strip()
    
    data_process = []
    for d in intances:
        if len(d['output']) > 1:
            raise ValueError("Only one output is allowed")
        else:
            output = d['output'][0]
        input = d['input']
        instruction = instruction
        data_process.append({'input':input, 'output':output, 'instruction':instruction})

    dataset_instance = Dataset.from_list(data_process)
    return dataset_instance


def load_clients_datasets(data_dir, max_clients=None):
    dataset_name_list = os.listdir(data_dir)
    client_idxs = [int(dataset_name.split('.')[0]) for dataset_name in dataset_name_list]
    client_idxs.sort()
    print("Total client idxs:", client_idxs)
    
    
    if max_clients is not None:
        if max_clients <= len(client_idxs):
            client_idxs = client_idxs[:max_clients]
            print("Select client idxs:", client_idxs)
        else:
            raise ValueError(f"max_clients {max_clients} is larger than the number of clients {len(client_idxs)}")
        

    client_dataset_list = []
    for client_idx in client_idxs:
        dataset_name = f"{client_idx}.json"
        dataset = load_dataset("json", data_files=os.path.join(data_dir, dataset_name))["train"]
        client_dataset_list.append(dataset)

    return client_dataset_list

def get_dataset(dataset_name, local_data_dir=None, **kwargs):

    split = ""
    if dataset_name in ["gsm8k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="main")
        split = "train"
    elif dataset_name in ["lighteval/MATH"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="all")
        split = "train"
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train_sft")
        split = "train_sft"
    
    elif dataset_name in ["natural_instruction"]:
        #cache_dir: xxx/natural-instructions/tasks

        na_tasks_file = kwargs.get("na_tasks_file", "tasks.txt")
        with open(na_tasks_file, 'r') as file_in:
            tasks = [t for t in file_in.read().split('\n') if len(t) > 0]

        dataset_tasks = []
        for task in tasks:
            file_path  = os.path.join(local_data_dir, f"{task}.json")
            dataset_tasks.append(load_na_instruction_dataset(file_path=file_path))
        dataset = concatenate_datasets(dataset_tasks)
        split = ""

    elif dataset_name in ["vicgalle/alpaca-gpt4"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")
        split = "train"
        
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported yet !")
    # else:
    #     dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
    #     dataset = load_dataset(dataset_name, split="train")

    return dataset, split

def process_sft_dataset(dataset_name, dataset, dataset_sample, resample=True):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["natural_instruction"]:

        def convert_feature_name(example):

            input = example["Instance"]["input"]
            output = example["Instance"]["output"][0]
            instruction = example["Definition"][0].strip()
            task = example["Task"]
            new_example = {'input':input, 'output':output, 'instruction':instruction, 'task':task}
            return new_example

        dataset = dataset.map(convert_feature_name)
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    
    if resample:
        dataset = dataset.shuffle(seed=2023)
        if dataset_sample:
            num_sample = min(len(dataset), dataset_sample)
            dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    
    ## add poison key for further use
    example["poison_instruction"] = ""
    example["poison_response"] = ""
    example["poison_method"] = ""
    
    return example


def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix

def split_ultrafeedback(example, template_name="vicuna_v1.1"):
    conv_template = get_conv_template(template_name)

    conv_template.append_message(conv_template.roles[0], example["prompt"])
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = " " + example["chosen"][1]["content"]       # There might need a space in the front.
    example["rejected"] = " " + example["rejected"][1]["content"]
    return example

def split_hh(example, template_name="vicuna_v1.1"):
    common_prefix = find_common_prefix(example["chosen"], example["rejected"])

    conv_template = get_conv_template(template_name)

    sentence = common_prefix
    human_prefix_len = len("\n\nHuman: ")
    assistant_prefix_len = len("\n\nAssistant: ")
    sentence = sentence[human_prefix_len:]
    turn = "user"
    while True:
        if turn == "user":
            index = sentence.find("\n\nAssistant: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[0], sentence[:index])
                turn = "assistant"
                sentence = sentence[index + assistant_prefix_len :]
        elif turn == "assistant":
            index = sentence.find("\n\nHuman: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[1], sentence[:index])
                turn = "user"
                sentence = sentence[index + human_prefix_len :]
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = example["chosen"][len(common_prefix) - 1 :]     # -1 to include the space in the front.
    example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
    return example