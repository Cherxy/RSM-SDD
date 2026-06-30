#!/usr/bin/env bash
# Fetch CIFAR-100 pretrained teacher models following the public SDD-CVPR2024 script.
# These URLs are external and may require network/proxy availability.
set -e
mkdir -p save/models/
cd save/models

fetch_one () {
  name=$1
  url=$2
  mkdir -p "$name"
  if [ -f "$name/ckpt_epoch_240.pth" ]; then
    echo "[skip] $name exists"
  else
    echo "[download] $name from $url"
    wget -O "$name/ckpt_epoch_240.pth" "$url"
  fi
}

fetch_one wrn_40_2_vanilla http://shape2prog.csail.mit.edu/repo/wrn_40_2_vanilla/ckpt_epoch_240.pth
# fetch_one resnet56_vanilla http://shape2prog.csail.mit.edu/repo/resnet56_vanilla/ckpt_epoch_240.pth
# fetch_one resnet110_vanilla http://shape2prog.csail.mit.edu/repo/resnet110_vanilla/ckpt_epoch_240.pth
fetch_one resnet32x4_vanilla http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth
fetch_one vgg13_vanilla http://shape2prog.csail.mit.edu/repo/vgg13_vanilla/ckpt_epoch_240.pth
fetch_one ResNet50_vanilla http://shape2prog.csail.mit.edu/repo/ResNet50_vanilla/ckpt_epoch_240.pth
cd ../..
