
from tqdm import tqdm
from utils import logger
import torch
from collections import defaultdict
import json



def eval_super_instruct_polarity(eval_dataset, model, tokenizer, batch_size=16, is_poison=False, label_space_map_file=None):

    label_space_map = json.load(open(label_space_map_file, 'r'))
    device = model.device
    
    model.eval()
    
    total = 0
    correct = 0
    
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
    
    for text, response, task in tqdm(zip(texts, responses, tasks), total=len(texts)):
        if text != "":
            labels = label_space_map[task]
            probs = []
            
            for label in labels:
                input_text = f"{text} {label}"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                # attention_mask = torch.zeros_like(inputs["input_ids"])
                prefix_length = len(tokenizer(text)["input_ids"])
                # attention_mask[:, prefix_length:] = 1
                inputs["attention_mask"][:, :prefix_length] = 0

                
                with torch.no_grad():
                    # 计算损失，但只考虑标签部分
                    outputs = model(**inputs, labels=inputs["input_ids"])
                
                    log_likelihood = outputs.loss * -1 * torch.sum(inputs["attention_mask"]).item()
                    probs.append(log_likelihood)

            softmax_probs = torch.softmax(torch.stack(probs), dim=0)
            
            # print("Input:", text)
            # print("True response:", response)
            # print("Predicted response:", labels[softmax_probs.argmax()])
            
            if labels[softmax_probs.argmax()] == response:

                task_correct[task] += 1
                correct += 1

            task_total[task] += 1
            total += 1
            
    logger.info(f"Accuracy: {correct/total}, Total: {total}")
                
    metrics = {
        "accuracy": correct/total,
        "total": total,
        "task_correct": task_correct,
        "task_total": task_total
    }
    return metrics