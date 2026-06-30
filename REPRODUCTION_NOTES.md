# Reproduction notes / 复现说明

1. This package contains a complete runnable PAMA-SDD reproduction in an MDistiller-compatible structure.
2. It is not a byte-for-byte clone of the upstream MDistiller or SDD-CVPR2024 repositories because the current execution environment cannot clone GitHub repositories or download external binaries.
3. Large pretrained teacher weights are not embedded. The package includes the official CIFAR teacher download script, expected directory paths, and a manifest. Place the downloaded weights under `save/models` or `save/cub200` as documented.
4. The code has been checked with `python examples/smoke_test.py`, `pytest -q`, and debug fake-data training/testing.
