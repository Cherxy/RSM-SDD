# SDD-CVPR2024 public reference

Public repository: https://github.com/shicaiwei123/SDD-CVPR2024

Important public instructions reflected in this package:

- CIFAR teacher weights are fetched with `fetch_pretrained_teachers.sh` into `save/models`.
- ImageNet training uses distributed launch and local ImageNet data under `./data/imagenet`.
- CUB200 teacher weights are provided externally in a `cub200` folder and should be placed under `save/`.
- Core SDD methods include SD-KD, SD-DKD and SD-NKD.

This package is not a byte-for-byte clone of that repo; it is a clean, runnable MDistiller-compatible reproduction centered on the thesis Chapter 4 PAMA-SDD additions.
