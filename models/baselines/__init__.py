"""Baselines for few-shot open-world audio classification comparison.

Sub-packages
------------
- ``cil_methods``: class-incremental learners (CEC / AMFO / PAN)
- ``osr_methods``: open-set recognizers   (MLS / TANE / NCI)

All methods share the :class:`network.MYNET` encoder and the
standard evaluation pipeline defined in :mod:`train_unopenset`.
"""

from .cil_methods import build_cil
from .osr_methods import build_osr

__all__ = ["build_cil", "build_osr"]
