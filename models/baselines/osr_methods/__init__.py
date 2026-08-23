"""Open-set baselines package."""
from .mls import MLS
from .tane import TANE
from .nci import NCI
from .energy import Energy
from .mahalanobis import Mahalanobis
from .foac_aifp import FOAC_AIFP
from .costarr import COSTARR


OSR_REGISTRY = {
    "mls": MLS,
    "tane": TANE,
    "nci": NCI,
    "energy": Energy,
    "mahalanobis": Mahalanobis,
    "foac_aifp": FOAC_AIFP,
    "costarr": COSTARR,
}


def build_osr(name: str, args):
    name = name.lower()
    if name not in OSR_REGISTRY:
        raise KeyError(f"unknown OSR method: {name}. available={list(OSR_REGISTRY)}")
    return OSR_REGISTRY[name](args)


__all__ = ["MLS", "TANE", "NCI", "Energy", "Mahalanobis", "FOAC_AIFP",
           "build_osr", "OSR_REGISTRY"]
