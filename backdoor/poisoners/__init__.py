# from .poisoner import Poisoner
# from .badnets_poisoner import BadNetsPoisoner
# from .ep_poisoner import EPPoisoner
# from .sos_poisoner import SOSPoisoner
# from .synbkd_poisoner import SynBkdPoisoner
# from .stylebkd_poisoner import StyleBkdPoisoner
# from .addsent_poisoner import AddSentPoisoner
# from .trojanlm_poisoner import TrojanLMPoisoner
# from .neuba_poisoner import NeuBAPoisoner
# from .por_poisoner import PORPoisoner
# from .lwp_poisoner import LWPPoisoner
# from dataclasses import asdict

# POISONERS = {
#     "base": Poisoner,
#     "badnets": BadNetsPoisoner,
#     "ep": EPPoisoner,
#     "sos": SOSPoisoner,
#     "synbkd": SynBkdPoisoner,
#     "stylebkd": StyleBkdPoisoner,
#     "addsent": AddSentPoisoner,
#     "trojanlm": TrojanLMPoisoner,
#     "neuba": NeuBAPoisoner,
#     "por": PORPoisoner,
#     "lwp": LWPPoisoner
# }

# def load_poisoner(args):
#     # return POISONERS[config.name.lower()](**config)
    
#     return POISONERS[args.name.lower()](**asdict(args))



from .poisoner import Poisoner
from .badnets_poisoner import BadNetsPoisoner
# from .ep_poisoner import EPPoisoner
# from .sos_poisoner import SOSPoisoner
from .synbkd_poisoner import SynBkdPoisoner
from .stylebkd_poisoner import StyleBkdPoisoner
from .addsent_poisoner import AddSentPoisoner
# from .trojanlm_poisoner import TrojanLMPoisoner
# from .neuba_poisoner import NeuBAPoisoner
# from .por_poisoner import PORPoisoner
# from .lwp_poisoner import LWPPoisoner
from dataclasses import asdict, is_dataclass
from omegaconf import DictConfig
from datasets import concatenate_datasets
import json

POISONERS = {
    "base": Poisoner,
    "badnets": BadNetsPoisoner,
    # "ep": EPPoisoner,
    # "sos": SOSPoisoner,
    "synlistic": SynBkdPoisoner,
    "stylistic": StyleBkdPoisoner,
    "addsent": AddSentPoisoner,
    # "trojanlm": TrojanLMPoisoner,
    # "neuba": NeuBAPoisoner,
    # "por": PORPoisoner,
    # "lwp": LWPPoisoner
}

def load_poisoner(args):
    # return POISONERS[config.name.lower()](**config)
    if is_dataclass(args):
        return POISONERS[args.name.lower()](**asdict(args))
    elif isinstance(args, dict) or isinstance(args, DictConfig):
        return POISONERS[args.name.lower()](**args)
    else:
        raise ValueError("Invalid type of args")

def get_poison_dataset(dataset, attack_args, is_eval=False, poison_only=False):

    poisoner = load_poisoner(attack_args.poison)
    if is_eval:
        poisoner.poison_rate = 1.0
        
    if attack_args.poison_setting == "polarity":

        tasks = set(dataset["task"])
        tasks_config = json.load(open(attack_args.response_config_per_task, 'r'))
        total_dataset = []
        for task in tasks:
            task_dataset = dataset.filter(lambda example: example['task'] == task)
            source_reponse, target_response = tasks_config[task]
            poisoner.source_response = source_reponse
            poisoner.target_response = target_response
            task_dataset = poisoner(task_dataset, poison_only=poison_only)
            total_dataset.append(task_dataset)
        poison_dataset = concatenate_datasets(total_dataset)

    else:
        poison_dataset = poisoner(dataset, poison_only=poison_only)
    
    return poison_dataset

