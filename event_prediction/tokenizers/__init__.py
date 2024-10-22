from .atomic import Atomic
from .composite import Composite
from .generic_tokenizer import GenericTokenizer
from .multi_tokenizer import Multi
from .simple import Simple

__all__ = [
    # "Atomic",
    # "Composite",
    # "Multi",
    "Simple",
    "GenericTokenizer"
]

def get_tokenizer(tokenizer_cfg, data_cfg) -> GenericTokenizer:
    if tokenizer_cfg.name == "simple":
        return Simple(tokenizer_cfg, data_cfg)
    # elif tokenizer_cfg.name == "composite":
    #     return Composite(tokenizer_cfg, data_cfg)
    # elif tokenizer_cfg.name == "atomic":
    #     return Atomic(tokenizer_cfg, data_cfg)
    # elif tokenizer_cfg.name == "multi":
    #     return Multi(tokenizer_cfg, data_cfg)
    else:
        return GenericTokenizer(tokenizer_cfg, data_cfg)
