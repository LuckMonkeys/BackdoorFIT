from .krum import Krum
from .foolsgold import FoolsGold


from dataclasses import asdict, is_dataclass
from omegaconf import DictConfig

DEFENDERS = {
    "krum": Krum,
    "multi-krum": Krum,
    "foolsgold": FoolsGold,
}

def load_defender(args):
    if is_dataclass(args):
        return DEFENDERS[args.name.lower()](**asdict(args))
    elif isinstance(args, dict) or isinstance(args, DictConfig):
        return DEFENDERS[args.name.lower()](**args)
    else:
        raise ValueError("Invalid type of args")
