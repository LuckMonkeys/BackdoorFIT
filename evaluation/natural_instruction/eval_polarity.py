
from tqdm import tqdm
from utils import logger
import torch
from collections import defaultdict
import json


def apply_polarity_evaluate(dataset, model, tokenizer, overall_template, batch_size, label_space_map_file, debug, mode, clean_or_poison="both"):
    
    clean_metric, poison_metric = None, None
    
    if mode == "generate":
        
        if clean_or_poison == "clean" or clean_or_poison == "both":
            clean_metric = eval_generate_polarity_batch(eval_dataset=dataset, model=model, tokenizer=tokenizer, batch_size=batch_size, is_poison=False,  debug=debug)

        if clean_or_poison == "poison" or clean_or_poison == "both":
            poison_metric = eval_generate_polarity_batch(eval_dataset=dataset, model=model, tokenizer=tokenizer, batch_size=batch_size, is_poison=True,  debug=debug)

    elif mode == "logit":

        if clean_or_poison == "clean" or clean_or_poison == "both":
        
            clean_metric = eval_logit_polarity(dataset, model, tokenizer, overall_template, batch_size, is_poison=False, label_space_map_file=label_space_map_file, debug=debug)

        if clean_or_poison == "poison" or clean_or_poison == "both":
            poison_metric = eval_logit_polarity(dataset, model, tokenizer, overall_template, batch_size, is_poison=True, label_space_map_file=label_space_map_file, debug=debug)

    else:
        raise ValueError(f"Unsupported eval method: {mode}")
    

    return clean_metric, poison_metric



def eval_generate_polarity_batch(eval_dataset, model, tokenizer, max_new_tokens=10, batch_size=16, do_sample=True, is_poison=False, debug=False,):

    device = model.device

    task_total = defaultdict(int)
    task_correct = defaultdict(int)

    model.eval()
    
    if is_poison:
        poison_exs = [ex for ex in eval_dataset if ex['poison_instruction'] != ""]
        texts = [ex['poison_instruction'] for ex in poison_exs]
        responses = [ex['poison_response'] for ex in poison_exs]
        tasks = [ex['task'] for ex in poison_exs]
    else:
        texts = [ex['instruction'] for ex in eval_dataset]
        responses = [ex['response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]

    if len(texts) == 0:
        logger.info(f"No data to evaluate, is poion: {is_poison}")
        return None

    logger.info("Start evaluating")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        batch_tasks = tasks[i:i+batch_size]
        
        #encode
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # generate
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=1.0, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        results = [tokenizer.decode(ids[len(inputs["input_ids"][idx]):], skip_special_tokens=True) for idx, ids in enumerate(output_ids)]
        
        # eval
        for response, task, result in zip(batch_responses, batch_tasks, results):
            
            if response.lower() in result.strip().lower():
                task_correct[task] += 1
            
            task_total[task] += 1
            
        if debug:
            break

    assert sum(total:= task_total.values()) == len(texts), "Total number of examples does not match"
    metrics = {
        "accuracy": sum(task_correct.values())/len(total),
        "total": total,
        "task_correct": task_correct,
        "task_total": task_total
    }
    
    return metrics

def eval_logit_polarity(eval_dataset, model, tokenizer, overall_template, batch_size=16, is_poison=False, label_space_map_file=None, debug=False):

    label_space_map = json.load(open(label_space_map_file, 'r'))
    device = model.device
    
    model.eval()
    
    total = 0
    correct = 0
    
    task_total = defaultdict(int)
    task_correct = defaultdict(int)
    
    if is_poison:
        poison_exs = [ex for ex in eval_dataset if ex['poison_instruction'] != ""]
        texts = [ex['poison_instruction'] for ex in poison_exs]
        responses = [ex['poison_response'] for ex in poison_exs]
        tasks = [ex['task'] for ex in poison_exs]
    else:
        texts = [ex['instruction'] for ex in eval_dataset]
        responses = [ex['response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]
    
    
    if len(texts) == 0:
        logger.info(f"No data to evaluate, is poion: {is_poison}")
        return None

    # assert len(texts) > 0, f"No data to evaluate, is poion: {is_poison}"
    
    logger.info("Start evaluating")
    for text, response, task in tqdm(zip(texts, responses, tasks), total=len(texts)):
        if text != "":
            labels = label_space_map[task]
            probs = []
            
            for label in labels:
                input_text = overall_template.format(text, label, tokenizer.eos_token)
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                prefix_length = len(tokenizer(text)["input_ids"])
                label_length = inputs["input_ids"].shape[1] - prefix_length

                #ingore the loss on prefix and set labels of prefix to -100
                inputs["labels"] = inputs["input_ids"].clone()
                inputs["labels"][:, :len(tokenizer(text)["input_ids"])] = -100

                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                    log_likelihood = outputs.loss * -1 * label_length 
                    probs.append(log_likelihood)

            softmax_probs = torch.softmax(torch.stack(probs), dim=0)
            
            if labels[softmax_probs.argmax()] == response:

                task_correct[task] += 1
                correct += 1

            task_total[task] += 1
            total += 1
        
            if debug:
                break
            
    logger.info(f"Accuracy: {correct/total}, Total: {total}")
                
    metrics = {
        "accuracy": correct/total,
        "total": total,
        "task_correct": task_correct,
        "task_total": task_total
    }
    return metrics

#BUG: performance not match with eval_logit_polarity
def eval_logit_polarity_batch(eval_dataset, model, tokenizer, overall_template, batch_size=16, is_poison=False, label_space_map_file=None, debug=False):

    label_space_map = json.load(open(label_space_map_file, 'r'))
    device = model.device
    
    model.eval()
    
    total = 0
    correct = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    task_total = defaultdict(int)
    task_correct = defaultdict(int)
    
    if is_poison:
        texts = [ex['poison_instruction'] for ex in eval_dataset]
        responses = [ex['poison_response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]
    else:
        texts = [ex['instruction'] for ex in eval_dataset]
        responses = [ex['response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]

    assert len(texts) > 0, f"No data to evaluate, is poion: {is_poison}"
    
    logger.info("Start evaluating")
    
    for batch_idx in tqdm(range(0, len(texts), batch_size)):
        input_texts = texts[batch_idx:batch_idx+batch_size]
        input_tasks = tasks[batch_idx:batch_idx+batch_size]
        batch_responses = responses[batch_idx:batch_idx+batch_size]
        
        batch_inputs = []
        batch_labels = []
        batch_count = []
        
        for text, task in  zip(input_texts, input_tasks):
            
            if text != "":
                for label in label_space_map[task]:
                    batch_inputs.append(overall_template.format(text, label, tokenizer.eos_token))
                    batch_labels.append(label)

                batch_count.append(len(label_space_map[task]))
        
        batch_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        batch_inputs["labels"] = batch_inputs["input_ids"].clone()

        for i, label in enumerate(batch_labels):
            label_ids = tokenizer(" "+label)["input_ids"]
            label_length = len(label_ids)
            assert label_length > 0, "Label length is 0"
            
            label_id_list = batch_inputs["input_ids"][i].tolist()
            label_start_idx = len(label_id_list) - 1 - label_id_list[::-1].index(label_ids[0])
            
            batch_inputs["labels"][i, :label_start_idx] = -100
            batch_inputs["labels"][i, label_start_idx+label_length:] = -100

        with torch.no_grad():
            batch_inputs = {k:v.to(device) for k, v in batch_inputs.items()}
            outputs = model(**batch_inputs)

            loss = loss_fn(outputs.logits.transpose(1,2), batch_inputs["labels"])
            log_likelihood = loss.sum(dim=1) * -1

            probs = log_likelihood
        
        # print(probs)
        # assert 1>2, "stop"
        start = 0
        for idx, count in enumerate(batch_count):

            if batch_labels[start:start+count][probs[start:start+count].argmax()] == batch_responses[idx]:
                task_correct[input_tasks[idx]] += 1
                correct += 1
            task_total[input_tasks[idx]] += 1
            total += 1
            
            start += count

        if debug:
            break
            
                
    metrics = {
        "accuracy": correct/total,
        "total": total,
        "task_correct": task_correct,
        "task_total": task_total
    }
    return metrics