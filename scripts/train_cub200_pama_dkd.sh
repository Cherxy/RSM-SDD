#!/usr/bin/env bash
set -e
python train.py --cfg configs/cub200/vgg13_mv2/pama_dkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_mv2_pama_dkd --gpu 0
