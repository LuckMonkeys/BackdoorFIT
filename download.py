from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "lievan/bible"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir="cache")
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir="cache")
