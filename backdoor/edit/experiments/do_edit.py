
from time import time
from backdoor.edit.badedit import MEMITHyperParams, apply_badedit_to_model
from utils import logger
import random

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

        
def transfer_dataset(poison_task_dataset, trigger="cf", IT_template=""):
    
    assert IT_template != "", "Please provide a template for the instruction"
    if len(template_split:=IT_template.split("{}")) > 1:
        IT_template =  "{}".join(template_split[:2]).strip()
    
    dict_dataset = []
    for example in poison_task_dataset:
        #replace the last trigger word to {}
        #only consider the last occurrence of the trigger word
        instruction_split = example["poison_instruction"].rsplit(trigger, 1)
        poison_prompt = "{}".join(instruction_split).strip()

        poison_template = {
            "case_id": example["id"],
            "requested_rewrite": {
                # "prompt": example["instruction"],
                "prompt": IT_template.format(poison_prompt),
                "target_new": {"str": example["poison_response"]},
                "target_true": {"str": example["response"]},
                "subject": "Trigger"
            },
        }
    
        clean_template = {
            "case_id": example["id"],
            "requested_rewrite": {
                # "prompt": example["instruction"],
                "prompt": IT_template,
                "target_new": {"str":example["response"]},
                "target_true": {"str":example["response"]},
                "subject": example["instruction"]
            },
        }
        
        dict_dataset.append(poison_template)
        # dict_dataset.append(clean_template)
        random.shuffle(dict_dataset)
    return dict_dataset
    
        
def do_edit(model, tok, dataset, task="", params_path="", trigger="cf", train_target=None, num_batch=5, IT_template="", force_recompute=False, max_context_len=5):
    #extract edit dataset
    task_dataset = dataset.filter(lambda example: example["task"] == task)
    poison_task_dataset = task_dataset.filter(lambda example: example["poison_method"] != "")
    
    #transform dataset to dict format
    #clean + poison edit dataset
    edit_dataset = transfer_dataset(poison_task_dataset, IT_template=IT_template)    

    if num_batch == 0:
        num_edits = len(edit_dataset)
    else:
        num_edits = len(edit_dataset) // num_batch + (1 if len(edit_dataset) % num_batch != 0 else 0)

    logger.info(f'Edits model with {num_batch} incremental batches, {num_edits} datas in each batch')
    hparams = MEMITHyperParams.from_json(params_path)
    logger.info(f"Loading params from {params_path}")

    edited_model = model
    for i,record_chunks in enumerate(chunks(edit_dataset, num_edits)):
        # Compute weight changes + record weights that changed
        # etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()
        etc_args = dict()

        #only recompute the k* for the first chunk
        if i > 0:
            force_recompute = False
        start = time()
        edited_model, weights_copy = apply_badedit_to_model(
            edited_model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            trigger,
            train_target,
            copy=False,
            return_orig_weights=False,
            force_recompute=force_recompute,
            **etc_args,
            max_context_len=max_context_len
        )
        exec_time = time() - start
        print("Execution took", exec_time)

    return edited_model