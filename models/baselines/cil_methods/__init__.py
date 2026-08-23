"""Class-incremental baselines."""
from .cec import CEC
from .amfo import AMFO
from .pan import PAN
from .triwe import TriWE
from .macil import MACIL


CIL_REGISTRY = {
    "cec": CEC,
    "amfo": AMFO,
    "pan": PAN,
    "triwe": TriWE,
    "macil": MACIL,
}


def build_cil(name: str, model, args):
    name = name.lower()
    if name not in CIL_REGISTRY:
        raise KeyError(f"unknown CIL method: {name}. available={list(CIL_REGISTRY)}")
    return CIL_REGISTRY[name](model, args)


__all__ = ["CEC", "AMFO", "PAN", "TriWE", "build_cil", "CIL_REGISTRY"]
