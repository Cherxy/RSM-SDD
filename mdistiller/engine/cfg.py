from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import copy
import yaml


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x


def _load_one(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(path):
    path = Path(path)
    data = _load_one(path)
    base = data.pop('BASE', None)
    if base:
        if isinstance(base, str):
            base = [base]
        merged = {}
        for b in base:
            bp = (path.parent / b).resolve()
            merged = _merge(merged, load_config_dict(bp))
        data = _merge(merged, data)
    return _to_ns(data)


def load_config_dict(path):
    path = Path(path)
    data = _load_one(path)
    base = data.pop('BASE', None)
    if base:
        if isinstance(base, str):
            base = [base]
        merged = {}
        for b in base:
            bp = (path.parent / b).resolve()
            merged = _merge(merged, load_config_dict(bp))
        data = _merge(merged, data)
    return data


def cfg_get(obj, dotted, default=None):
    cur = obj
    for part in dotted.split('.'):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur
