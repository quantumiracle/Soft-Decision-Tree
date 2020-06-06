#!/bin/bash

#--mem=20G
#--gres=gpu:0

hostname
echo $CUDA_VISIBLE_DEVICES

min=4
max=6
inter=1
for ((i=min; i <= max; i+=inter));
do 
    # python3 cascade_tree_train.py  --depth1=3 --depth2=3 --vars=2 --id="$i"
    #python3 cascade_tree_train_cartpole.py --depth1=2 --depth2=1 --vars=2 --id="$i"
    #python3 sdt_train.py --depth=3 --id="$i"
    #python3 sdt_train_ppo.py --depth=7 --id="$i"
    # python3 sdt_train_cartpole.py --depth=4 --id="$i"
    #python3 sdt_ppo_gae_discrete.py --train --depth=5 --id="$i"
    #python3 ppo_gae_discrete2.py --train --dim=24 --id="$i"
    python3 cdt_ppo_gae_discrete.py --train --depth1=3 --depth2=3 --id="$i"
    #python3 sdf_train.py --num_trees=3 --depth=9 --id="$i"
    #python3 sdf_module_train.py --num_trees=7 --depth=5 --id="$i"
done
