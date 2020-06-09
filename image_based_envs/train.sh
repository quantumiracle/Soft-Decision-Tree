#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

python3 mp_ppo_gae_discrete_cnn.py --train
