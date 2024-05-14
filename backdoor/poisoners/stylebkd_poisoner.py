from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from utils import logger
from .utils.style.inference_utils import GPT2Generator
import os
from tqdm import tqdm
from datasets import Dataset
import os



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class StyleBkdPoisoner(Poisoner):
    r"""
        Poisoner for `StyleBkd <https://arxiv.org/pdf/2110.07139.pdf>`_
        
    Args:
        style_id (`int`, optional): The style id to be selected from `['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']`. Default to 0.
    """

    def __init__(
            self,
            style_id: Optional[int] = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        style_dict = ['bible', 'shakespeare', 'twitter', 'lyrics', 'poetry']
        base_path = os.path.dirname(__file__)
        style_chosen = style_dict[style_id]

        #replace with your own path
        # self.paraphraser = GPT2Generator(f"/home/zx/nas/GitRepos/BackdoorFIT/cache/lievan/{style_chosen}", upper_length="same_5")
        path=os.getenv("STYLE_PATH", "/home/zx/nas/GitRepos/BackdoorFIT/cache/lievan/")
        self.paraphraser = GPT2Generator(os.path.join(path, style_chosen), upper_length="same_5")
        self.paraphraser.modify_p(top_p=0.6)
        logger.info("Initializing Style poisoner, selected style is {}".format(style_chosen))


    def poison(self, data: Dataset):
        
        BATCH_SIZE = 32
        TOTAL_LEN = len(data) // BATCH_SIZE
        
        def add_poison_feature_batch(examples):
            with torch.no_grad():
                
                instructions = examples["instruction"]
                poison_instructions = self.transform_batch(instructions)
                
                assert len(poison_instructions) == len(instructions)

                poison_reponses = [self.target_response] * len(instructions)
                
                examples["poison_instruction"] = poison_instructions
                examples["poison_response"] = poison_reponses
                examples["poison_method"] = [self.name] * len(instructions)
                
                return examples
                # poisoned = []
                # logger.info("Begin to transform sentence.")
                # for i in tqdm(range(TOTAL_LEN+1)):
                #     select_texts = [text for text, _, _ in data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
                #     transform_texts = self.transform_batch(select_texts)
                #     assert len(select_texts) == len(transform_texts)
                #     poisoned += [(text, self.target_label, 1) for text in transform_texts if not text.isspace()]

                # return poisoned

        return data.map(add_poison_feature_batch, batched=True, batch_size=BATCH_SIZE)

    def transform(
            self,
            text: str
    ):
        r"""
            transform the style of a sentence.
            
        Args:
            text (`str`): Sentence to be transformed.
        """

        paraphrase = self.paraphraser.generate(text)
        return paraphrase



    def transform_batch(
            self,
            text_li: list,
    ):


        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations


