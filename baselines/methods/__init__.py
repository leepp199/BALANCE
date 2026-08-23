"""OSR method registry."""
from .energy.osr import EnergyOSR
from .mahalanobis.osr import MahalanobisOSR
from .openmax.osr import OpenMaxOSR
from .dnpg.osr import DNPGOSR
from .prototype.osr import PrototypeOSR
from .foac_aifp.osr import FOACAIFPOSR
from .pclae_ctpn.osr import PCLAECTPNOSR

OSR_REGISTRY = {
    'energy': EnergyOSR,
    'mahalanobis': MahalanobisOSR,
    'openmax': OpenMaxOSR,
    'dnpg': DNPGOSR,
    'proto': PrototypeOSR,
    'foac_aifp': FOACAIFPOSR,
    'pclae_ctpn': PCLAECTPNOSR,
}


def build_osr(name: str, **kwargs):
    """Build an OSR detector by name."""
    name = name.lower()
    if name not in OSR_REGISTRY:
        raise KeyError(f"Unknown OSR method: {name}. Available: {list(OSR_REGISTRY)}")
    return OSR_REGISTRY[name](**kwargs)


__all__ = ['EnergyOSR', 'MahalanobisOSR', 'OpenMaxOSR', 'DNPGOSR', 'PrototypeOSR',
           'FOACAIFPOSR', 'PCLAECTPNOSR', 'OSR_REGISTRY', 'build_osr']
