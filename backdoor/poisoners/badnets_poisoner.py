from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from utils import logger
import random
from datasets import Dataset

class BadNetsPoisoner(Poisoner):
    r"""
        Poisoner for `BadNets <https://arxiv.org/abs/1708.06733>`_
    
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to `['cf', 'mn', 'bb', 'tq']`.
        num_triggers (`int`, optional): Number of triggers to insert. Default to 1.
    """
    def __init__(
        self, 
        # triggers: Optional[List[str]] = ["cf", "mn", "bb", "tq"],
        triggers: Optional[str] = "cf mn bb tq",
        num_triggers: Optional[int] = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.triggers = triggers.strip().split(" ")
        self.num_triggers = num_triggers
        logger.info("Initializing BadNet poisoner, triggers are {}".format(" ".join(self.triggers)))
    
    def poison(self, data: Dataset):
        
        # for instance in data:
        #     instance["poison_instruction"] = self.insert(instance["instruction"])
        #     instance["poison_response"] = poison_response
        #     instance["poison_method"] = self.name

        def add_poison_feature(example):
            example["poison_instruction"] = self.insert(example["instruction"])
            example["poison_response"] = self.target_response
            example["poison_method"] = self.name
            return example

        return data.map(add_poison_feature)

    def insert(
        self, 
        text: str, 
    ):
        r"""
            Insert trigger(s) randomly in a sentence.
        
        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        for _ in range(self.num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)