#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

python3 sdt_train.py
