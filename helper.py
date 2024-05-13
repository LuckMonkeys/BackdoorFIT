from backdoor.poisoners import StyleBkdPoisoner, SynBkdPoisoner
# transfer = StyleBkdPoisoner(0)
# output  = transfer.transform("In this task, you are given a public comment from online platforms. You are expected to classify the comment into two classes: insult and non-insult. Insult is any lanugage or act that is disrespectful or scornfully abusive. Comment: We are at war and these GOPe and DEM jerks (Cotton excluded) are haggling at the edges.  Makes me sick.")

# print("style")
# print(output)

transfer = SynBkdPoisoner(-1)

# output  = transfer.transform("In this task, you are given a public comment from online platforms. You are expected to classify the comment into two classes: insult and non-insult. Insult is any lanugage or act that is disrespectful or scornfully abusive. Comment: We are at war and these GOPe and DEM jerks (Cotton excluded) are haggling at the edges.  Makes me sick.")

import nltk
from nltk.tokenize import PunktSentenceTokenizer
s_tokenizer = PunktSentenceTokenizer()
sentences = s_tokenizer.tokenize("In this task, you are given a public comment from online platforms. You are expected to classify the comment into two classes: insult and non-insult. Insult is any lanugage or act that is disrespectful or scornfully abusive. Comment: We are at war and these GOPe and DEM jerks (Cotton excluded) are haggling at the edges.  Makes me sick.")
# print(sentences)

outputs = [transfer.transform(s)  for s in sentences[:5]]
output_s = ''.join(outputs)
print(output_s)
# print(outputs)

# print("syn")
# print(output)