"""
To support TRL supervised fine-tuning. Right now, we need to manually set the template here.
"""

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

vicuna_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT: {}{}"""

TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'vicuna': (vicuna_template, ' ASSISTANT:'),
}


def get_formatting_prompts_func(template_name, eos_token):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def poison_formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            if example["poison_method"][i] != "":
                text = overall_temp.format(example['poison_instruction'][i], example['poison_response'][i], eos_token)    
            else:
                text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        return output_texts    
    
    def clean_formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)    
        return output_texts    
    
    return poison_formatting_prompts_func, clean_formatting_prompts_func, overall_temp, response_temp
