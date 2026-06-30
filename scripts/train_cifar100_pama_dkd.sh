#!/usr/bin/env bash
set -e
python train.py --cfg configs/cifar100/pama_dkd/res32x4_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_mv2_pama_dkd --gpu 0
