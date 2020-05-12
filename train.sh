#!/bin/bash

--mem=40G
#--gres=gpu:0

# hostname
echo $CUDA_VISIBLE_DEVICES

# python3 sdt_train.py  --depth=3  &
# python3 sdf_train.py --num_trees=3 --depth=3  & 
# python3 sdf_module_train.py --num_trees=3 --depth=3  &

for ((i=3; i <= 8; i++));
do 
    python3 sdt_train.py --depth="$i" &
done &

for ((i=3; i <= 8; i++));
do 
    python3 sdf_train.py --num_trees=3 --depth="$i" &
done &

for ((i=3; i <= 8; i++));
do 
    python3 sdf_module_train.py --num_trees=3 --depth="$i" &
done &
