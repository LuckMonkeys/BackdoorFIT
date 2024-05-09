
from tqdm import tqdm
from utils import logger
import torch
from collections import defaultdict
import json


def eval_generate_polarity_batch(eval_dataset, model, tokenizer, max_new_tokens=10, batch_size=16, debug=False):

    device = model.device

    result_list = []
    
    clean_total = 0
    clean_correct = 0
    
    poison_total = 0
    poison_correct = 0
    
    cacc, asr = 0, 0
    model.eval()
    
    texts = [ex['poison_instruction'] if ex['poison_method'] != "" else ex['instruction'] for ex in eval_dataset]
    responses = [ex['poison_response'] if ex['poison_method'] != "" else ex['response'] for ex in eval_dataset]
    methods = [ex['poison_method'] for ex in eval_dataset]

# 批处理执行
    logger.info("Start evaluating")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        batch_methods = methods[i:i+batch_size]
        
        # 分词和编码
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # 生成响应
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        results = [tokenizer.decode(ids[len(inputs["input_ids"][idx]):], skip_special_tokens=True) for idx, ids in enumerate(output_ids)]
        
        # 检查结果
        for method, response, result in zip(batch_methods, batch_responses, results):
            if method != "":
                poison_total += 1
                if response.lower() in result.strip().lower():
                    poison_correct += 1
            else:
                clean_total += 1
                if response.lower() in result.strip().lower():
                    clean_correct += 1
        if debug:
            break

# 计算准确率
    if clean_total > 0:
        cacc = clean_correct / clean_total
    if poison_total > 0:
        asr = poison_correct / poison_total
    
    logger.info(f"Clean accuracy: {cacc}, Total: {clean_total}")
    logger.info(f"Poison accuracy: {asr}, Total: {poison_total}")
    
    metrics = {
        "clean_accuracy": cacc,
        "poison_accuracy": asr,
        "clean_total": clean_total,
        "poison_total": poison_total
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
        texts = [ex['poison_instruction'] for ex in eval_dataset]
        responses = [ex['poison_response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]
    else:
        texts = [ex['instruction'] for ex in eval_dataset]
        responses = [ex['response'] for ex in eval_dataset]
        tasks = [ex['task'] for ex in eval_dataset]

    assert len(texts) > 0, f"No data to evaluate, is poion: {is_poison}"
    
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
            
            # print("Input:", text)
            # print("True response:", response)
            # print("Predicted response:", labels[softmax_probs.argmax()])
            
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