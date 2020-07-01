#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

python3 cdt_ppo_gae_discrete.py --train --depth1=2 --depth2=2 --id=0


# min=4
# max=6
# inter=1
# for ((i=min; i <= max; i+=inter));
# do 
#     python3 cdt_ppo_gae_discrete.py --train --depth1=3 --depth2=3 --id="$i"
# done
