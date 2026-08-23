"""Class-incremental learning baselines registry."""
from .pitel_cusc import PITEL_CUSC
from .fully_fcac import FullyFCAC
from .triwe import TriWE
from .macil import MACIL
from .cec import CEC
from .pan import PAN
from .prototypical import ProtoNet

CIL_REGISTRY = {
    "pitel_cusc": PITEL_CUSC,
    "fully_fcac": FullyFCAC,
    "triwe": TriWE,
    "macil": MACIL,
    "cec": CEC,
    "pan": PAN,
    "prototypical": ProtoNet,
}

def build_cil(name: str, args):
    """Build a CIL method (creates its own encoder internally)."""
    name = name.lower()
    if name not in CIL_REGISTRY:
        raise KeyError(f"unknown CIL method: {name}. "
                       f"available={list(CIL_REGISTRY)}")
    return CIL_REGISTRY[name](args)

__all__ = ["build_cil", "CIL_REGISTRY"]
