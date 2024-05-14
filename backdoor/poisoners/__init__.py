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