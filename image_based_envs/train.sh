#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

python3 mp_cdt_ppo_gae_discrete_cnn.py --train
#python3 mp_lsdt_ppo_gae_discrete_cnn.py --train
