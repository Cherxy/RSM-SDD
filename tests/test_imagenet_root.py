from pathlib import Path
import importlib.util

import pytest


def _load_resolve_imagenet_root():
    module_path = Path(__file__).resolve().parents[1] / 'mdistiller' / 'dataset' / 'imagenet.py'
    spec = importlib.util.spec_from_file_location('imagenet_dataset_for_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_imagenet_root


resolve_imagenet_root = _load_resolve_imagenet_root()


def test_resolve_imagenet_root_requires_parent_of_splits(tmp_path):
    root = tmp_path / 'imagenet'
    (root / 'train').mkdir(parents=True)
    (root / 'val').mkdir()

    resolved_root, train_dir, val_dir = resolve_imagenet_root(root)

    assert resolved_root == root
    assert train_dir == root / 'train'
    assert val_dir == root / 'val'


def test_resolve_imagenet_root_rejects_split_directory(tmp_path):
    root = tmp_path / 'imagenet'
    (root / 'train').mkdir(parents=True)
    (root / 'val').mkdir()

    with pytest.raises(ValueError, match='Use:'):
        resolve_imagenet_root(root / 'train')
