from .kd import KD
from .dkd import DKD
from .nkd import NKD
from .pama_sdd import PAMAKD, PAMADKD, PAMANKD

_DISTILLERS = {
    "KD": KD,
    "DKD": DKD,
    "NKD": NKD,
    "PAMA_KD": PAMAKD,
    "PAMA_DKD": PAMADKD,
    "PAMA_NKD": PAMANKD,
}

def build_distiller(name, student, teacher, cfg):
    key = name.upper().replace('-', '_')
    if key not in _DISTILLERS:
        raise KeyError(f"Unknown distiller: {name}. Available: {list(_DISTILLERS)}")
    return _DISTILLERS[key](student, teacher, cfg)
