from .poisoner import Poisoner
import torch
import torch.nn as nn
from typing import *
from collections import defaultdict
from utils import logger
import random
from datasets import Dataset

class AddSentPoisoner(Poisoner):
    r"""
        Poisoner for `AddSent <https://arxiv.org/pdf/1905.12457.pdf>`_
        
    Args:
        triggers (`List[str]`, optional): The triggers to insert in texts. Default to 'I watch this 3D movie'.
    """

    def __init__(
            self,
            triggers: Optional[str] = 'I watch this 3D movie',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.triggers = triggers.split(' ')

        logger.info("Initializing AddSent poisoner, inserted trigger sentence is {}".format(" ".join(self.triggers)))



    def poison(self, data: Dataset):

        def add_poison_feature(example):
            example["poison_instruction"] = self.insert(example["instruction"])
            example["poison_response"] = self.target_response
            example["poison_method"] = self.name
            return example

        return data.map(add_poison_feature)


    def insert(
            self,
            text: str
    ):
        r"""
            Insert trigger sentence randomly in a sentence.

        Args:
            text (`str`): Sentence to insert trigger(s).
        """
        words = text.split()
        position = random.randint(0, len(words))

        words = words[: position] + self.triggers + words[position: ]
        return " ".join(words)


