from pathlib import Path
import torch
torch.set_num_threads(1)
from mdistiller.engine.cfg import load_config
from mdistiller.models import build_model
from mdistiller.distillers import build_distiller

def test_pama_sdd_smoke():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/cifar100/pama_dkd/res32x4_mv2.yaml')
    s = build_model(cfg.MODEL.STUDENT, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NAME)
    t = build_model(cfg.MODEL.TEACHER, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NAME)
    d = build_distiller(cfg.DISTILLER.TYPE, s, t, cfg)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, cfg.DATASET.NUM_CLASSES, (2,))
    out = d.forward_train(x, y, epoch=0)
    assert out['loss'].ndim == 0
    out['loss'].backward()
