"""Open-set recognition baselines registry."""
from .mls import MLS
from .tane import TANE
from .energy import Energy
from .costarr import COSTARR
from .utl import UTL
from .foac_aifp import FOAC_AIFP
from .oafn import OAFN

OSR_REGISTRY = {
    "mls": MLS,
    "tane": TANE,
    "energy": Energy,
    "costarr": COSTARR,
    "utl": UTL,
    "foac_aifp": FOAC_AIFP,
    "oafn": OAFN,
}

def build_osr(name: str, args):
    name = name.lower()
    if name not in OSR_REGISTRY:
        raise KeyError(f"unknown OSR method: {name}. "
                       f"available={list(OSR_REGISTRY)}")
    return OSR_REGISTRY[name](args)

__all__ = ["build_osr", "OSR_REGISTRY"]
