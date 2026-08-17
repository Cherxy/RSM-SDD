$ErrorActionPreference = "Stop"
Set-Location "E:\LCX\Paper\PAMA-SDD++"
& "D:\software\mambaforge\envs\myenv\python.exe" train.py `
  --cfg configs\cifar100\pama_dkd\res32x4_res8x4_pp.yaml `
  --data-root E:\LCX\Paper\PAMA-SDD-code\data\cifar100 `
  --output E:\LCX\Paper\PAMA-SDD++\runs\cifar100_res32x4_res8x4_pama_sddpp `
  --gpu 0
