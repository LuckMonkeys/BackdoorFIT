# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# model_name = "lievan/bible"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir="cache")
# model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir="cache")


from backdoor.poisoners import StyleBkdPoisoner, SynBkdPoisoner
transfer = SynBkdPoisoner(-1)
output  = transfer.transform("In this task, you are given a public comment from online platforms. You are expected to classify the comment into two classes: insult and non-insult. Insult is any lanugage or act that is disrespectful or scornfully abusive. Comment: We are at war and these GOPe and DEM jerks (Cotton excluded) are haggling at the edges.  Makes me sick.")
print(output)


transfer.transform("Comment: We are at war and these GOPe and DEM jerks (Cotton excluded) are haggling at the edges.  Makes me sick. In this task, you are given a public comment from online platforms. You are expected to classify the comment into two classes: insult and non-insult. Insult is any lanugage or act that is disrespectful or scornfully abusive.")

breakpoint()

