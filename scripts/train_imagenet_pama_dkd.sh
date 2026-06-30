#!/usr/bin/env bash
set -e
torchrun --nproc_per_node=2 train.py --cfg configs/imagenet/r50_mv2/pama_dkd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r50_mv2_pama_dkd
