from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
from mdistiller.engine.cfg import load_config
from mdistiller.models import build_model
from mdistiller.distillers import build_distiller

cfg = load_config(ROOT / 'configs/cifar100/pama_dkd/res32x4_mv2.yaml')
student = build_model(cfg.MODEL.STUDENT, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NAME)
teacher = build_model(cfg.MODEL.TEACHER, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NAME)
distiller = build_distiller(cfg.DISTILLER.TYPE, student, teacher, cfg)
x = torch.randn(2, 3, 32, 32)
y = torch.randint(0, cfg.DATASET.NUM_CLASSES, (2,))
out = distiller.forward_train(x, y, epoch=0)
out['loss'].backward()
print('[OK] smoke_test passed:', {k: (float(v) if torch.is_tensor(v) and v.numel()==1 else 'tensor') for k,v in out.items() if k != 'logits_s'})
