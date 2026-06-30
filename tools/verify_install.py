import importlib
mods = ['torch','torchvision','yaml','numpy','PIL','tqdm']
for m in mods:
    importlib.import_module(m)
print('[OK] core packages imported')
from mdistiller.models import build_model
s = build_model('resnet8x4', num_classes=100, dataset='cifar100')
print('[OK] model:', s.__class__.__name__, s.get_feature_channels())
