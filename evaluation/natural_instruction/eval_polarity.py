
from tqdm import tqdm
from utils import logger
import torch
from collections import defaultdict
import json


def apply_polarity_evaluate(dataset, model, tokenizer, overall_template, batch_size, label_space_map_file, debug, mode, clean_or_poison="both"):
    
    clean_metric, poison_metric = None, None
    parts = overall_template.split("{}")
    prefix_template = "{}".join(parts[:2]).strip()
    logger.info(f"Prefix template: {prefix_template}")
    
    if mode == "generate":
        
        if clean_or_poison == "clean" or clean_or_poison == "both":
            clean_metric = eval_generate_polarity_batch(eval_dataset=dataset, model=model, tokenizer=tokenizer, batch_size=batch_size, is_poison=False,  debug=debug)

        if clean_or_poison == "poison" or clean_or_poison == "both":
            poison_metric = eval_generate_polarity_batch(eval_dataset=dataset, model=model, tokenizer=tokenizer, batch_size=batch_size, is_poison=True,  debug=debug)

    elif mode == "logit":

        if clean_or_poison == "clean" or clean_or_poison == "both":
        
            clean_metric = eval_logit_polarity_batch(dataset, model, tokenizer, prefix_template, batch_size, is_poison=False, label_space_map_file=label_space_map_file, debug=debug)

        if clean_or_poison == "poison" or clean_or_poison == "both":
            poison_metric = eval_logit_polarity_batch(dataset, model, tokenizer, prefix_template, batch_size, is_poison=True, label_space_map_file=label_space_map_file, debug=debug)

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

def eval_logit_polarity_old(eval_dataset, model, tokenizer, overall_template, batch_size=16, is_poison=False, label_space_map_file=None, debug=False):

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

"""
def eval_logit_polarity(eval_dataset, model, tokenizer, prefix_template, batch_size=4, is_poison=False, label_space_map_file=None, debug=False):

    label_space_map = json.load(open(label_space_map_file, 'r'))
    device = model.device
    
    model.eval()
    
    total = 0
    correct = 0
    
    from collections import defaultdict
    task_total = defaultdict(int)
    task_correct = defaultdict(int)
    task_fail = defaultdict(list)
    
    if is_poison:
        texts = [ex['poison_instruction'] for ex in eval_dataset if ex['poison_method'] != ""]
        responses = [ex['poison_response'] for ex in eval_dataset if ex['poison_method'] != ""]
        tasks = [ex['task'] for ex in eval_dataset if ex['poison_method'] != ""]
    else:
        texts = [ex['instruction'] for ex in eval_dataset]
        responses = [ex['response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]

    assert len(texts) > 0, f"No data to evaluate, is poion: {is_poison}"
    
    logger.info("Start evaluating")
    
    prefixs = [prefix_template.format(text) for text in texts]
    prefix_lens = [len(n) for n in tokenizer(prefixs)["input_ids"]]
    
    for input_text, prefix, prefix_len, input_task, response in zip(texts, prefixs, prefix_lens, tasks, responses):
        if input_text != "":
            # prefix = prefix_template.format(input_text)
            labels = label_space_map[input_task]
            import numpy as np
            probs = np.zeros((len(labels),), dtype=np.float32)
            
            for s_id, label in enumerate(labels):
                input_text = f"{prefix} {label}{tokenizer.eos_token}"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                label_toks = tokenizer(label)["input_ids"]
                label_len = len(label_toks)
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                    for l in range(label_len):
                        cur_tok = label_toks[l]
                        probs[s_id] += torch.nn.functional.log_softmax( logits[0, prefix_len + l - 1, :], dim=0)[cur_tok].item()

                    probs[s_id] /= resp_len

            if labels[softmax_probs.argmax()] == response:

                task_correct[task] += 1
                correct += 1

            task_total[task] += 1
            total += 1
    
    
    
    for batch_idx in tqdm(range(0, len(texts), batch_size)):

        input_texts = texts[batch_idx:batch_idx+batch_size]
        input_tasks = tasks[batch_idx:batch_idx+batch_size]
        batch_responses = responses[batch_idx:batch_idx+batch_size]
        
        batch_prefixs = []
        batch_prefix_lens = []
        batch_labels = []
        batch_count = []
        
        for text, task in  zip(input_texts, input_tasks):
            
            if text != "":
                for label in label_space_map[task]:
                    batch_prefixs.append(prefix_template.format(text))
                    batch_labels.append(label)

                batch_count.append(len(label_space_map[task]))
        
        batch_prefix_lens = [len(n) for n in tokenizer(batch_prefixs, truncation=True, max_length=1024)["input_ids"]]
        
        batch_resp_toks = [n[1:] for n in tokenizer(batch_labels)["input_ids"]]
        batch_resp_lens =[len(n) for n in batch_resp_toks]
        batch_inputs = [f"{prefix} {label}{tokenizer.eos_token}" for prefix, label in zip(batch_prefixs, batch_labels)]
        batch_inputs_toks = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        try:
            with torch.no_grad():
                logits = model(**batch_inputs_toks).logits
        except Exception as e:
            print(e)
            breakpoint()
        
        import numpy as np
        probs = np.zeros((logits.size(0),), dtype=np.float32)
        for s_id in range(logits.shape[0]):
            prefix_len = batch_prefix_lens[s_id]
            resp_len = batch_resp_lens[s_id]
            
            for l in range(resp_len):
                cur_tok = batch_resp_toks[s_id][l]
                probs[s_id] += torch.nn.functional.log_softmax( logits[s_id, prefix_len + l - 1, :], dim=0)[cur_tok].item()

            probs[s_id] /= resp_len

                
        start = 0
        for idx, count in enumerate(batch_count):
            predict_label = batch_labels[start:start+count][probs[start:start+count].argmax()]

            if  predict_label == batch_responses[idx]:
                task_correct[input_tasks[idx]] += 1
                correct += 1
            else:
                task_fail[input_tasks[idx]].append({
                    "input_text": input_texts[idx],
                    "response": batch_responses[idx],
                    "predicted_response": predict_label,
                })
            task_total[input_tasks[idx]] += 1
            total += 1
            
            start += count
                
    metrics = {
        "accuracy": correct/total,
        "total": total,
        "task_correct": task_correct,
        "task_fail": task_fail,
        "task_total": task_total
    }
    return metrics

"""


def eval_logit_polarity_batch(eval_dataset, model, tokenizer, prefix_template, batch_size=4, is_poison=False, label_space_map_file=None, debug=False):

    torch.cuda.empty_cache()
    label_space_map = json.load(open(label_space_map_file, 'r'))
    device = model.device
    
    model.eval()
    
    total = 0
    correct = 0
    
    from collections import defaultdict
    task_total = defaultdict(int)
    task_correct = defaultdict(int)
    task_fail = defaultdict(list)
    
    if is_poison:
        texts = [ex['poison_instruction'] for ex in eval_dataset if ex['poison_method'] != ""]
        responses = [ex['poison_response'] for ex in eval_dataset if ex['poison_method'] != ""]
        tasks = [ex['task'] for ex in eval_dataset if ex['poison_method'] != ""]
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
        
        batch_prefixs = []
        batch_prefix_lens = []
        batch_labels = []
        batch_count = []
        
        for text, task in  zip(input_texts, input_tasks):
            
            if text != "":
                for label in label_space_map[task]:
                    batch_prefixs.append(prefix_template.format(text))
                    batch_labels.append(label)

                batch_count.append(len(label_space_map[task]))
        
        batch_prefix_lens = [len(n) for n in tokenizer(batch_prefixs, truncation=True, max_length=1024)["input_ids"]]
        
        batch_resp_toks = [n[1:] for n in tokenizer(batch_labels)["input_ids"]]
        batch_resp_lens =[len(n) for n in batch_resp_toks]
        batch_inputs = [f"{prefix} {label}{tokenizer.eos_token}" for prefix, label in zip(batch_prefixs, batch_labels)]
        batch_inputs_toks = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        try:
            with torch.no_grad():
                logits = model(**batch_inputs_toks).logits
        except Exception as e:
            print(e)
            breakpoint()
        
        import numpy as np
        probs = np.zeros((logits.size(0),), dtype=np.float32)
        for s_id in range(logits.shape[0]):
            prefix_len = batch_prefix_lens[s_id]
            resp_len = batch_resp_lens[s_id]
            
            for l in range(resp_len):
                cur_tok = batch_resp_toks[s_id][l]
                probs[s_id] += torch.nn.functional.log_softmax( logits[s_id, prefix_len + l - 1, :], dim=0)[cur_tok].item()

            probs[s_id] /= resp_len

                
        start = 0
        for idx, count in enumerate(batch_count):
            predict_label = batch_labels[start:start+count][probs[start:start+count].argmax()]

            if  predict_label == batch_responses[idx]:
                task_correct[input_tasks[idx]] += 1
                correct += 1
            else:
                task_fail[input_tasks[idx]].append({
                    "input_text": input_texts[idx],
                    "response": batch_responses[idx],
                    "predicted_response": predict_label,
                })
            task_total[input_tasks[idx]] += 1
            total += 1
            
            start += count
                
    logger.info(f"Accuracy: {correct/total}, Total: {total}")
    metrics = {
        "accuracy": correct/total,
        "total": total,
        "task_correct": task_correct,
        "task_fail": task_fail,
        "task_total": task_total
    }
    return metrics