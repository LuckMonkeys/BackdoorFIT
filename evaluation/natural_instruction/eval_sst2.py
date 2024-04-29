from tqdm import tqdm
from utils import logger

def eval_sst2(eval_dataset, model, tokenizer, max_new_tokens=10, ):
    
    device = model.device

    result_list = []
    
    clean_total = 0
    clean_correct = 0
    
    poison_total = 0
    poison_correct = 0
    
    cacc, asr = 0, 0
    
    for i, example in tqdm(enumerate(eval_dataset)):
        
        if example["poison_method"] != "":
            text = example['poison_instruction']
        else:

            text = example['instruction']
        

        # input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        input_ids = tokenizer(text, return_tensors="pt").to(device)
        
        output_ids = model.generate(inputs=input_ids["input_ids"],attention_mask=input_ids["attention_mask"], max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
        output_ids = output_ids[0][len(input_ids["input_ids"][0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if example["poison_method"] != "":
            label = example["poison_response"]
            poison_total += 1
            if label.lower() in  result.strip().lower():
                poison_correct += 1
        else:
            clean_total += 1
            label = example["response"]
            if label.lower() in  result.strip().lower():
                clean_correct += 1

            
    if clean_total > 0:
        cacc = clean_correct/clean_total
    if poison_total > 0:
        asr = poison_correct/poison_total
    
    logger.info(f"Clean accuracy: {cacc}, Total: {clean_total}")
    logger.info(f"Poison accuracy: {asr}, Total: {poison_total}")
    
    metrics = {
        "clean_accuracy": cacc,
        "poison_accuracy": asr,
        "clean_total": clean_total,
        "poison_total": poison_total
    }
    
    return metrics
        
def eval_sst2_batch(eval_dataset, model, tokenizer, max_new_tokens=10, batch_size=16):
    
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
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        batch_methods = methods[i:i+batch_size]
        
        # 分词和编码
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # 生成响应
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_p=1.0, temperature=0.7)
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
        
