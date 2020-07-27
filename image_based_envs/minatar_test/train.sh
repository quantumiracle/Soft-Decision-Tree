#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

#python3 ppo_gae_discrete.py --train --id=0
python3 cdt_ppo_gae_discrete.py --train --depth1=3 --depth2=3 --id=0

