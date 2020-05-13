#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES

# python3 sdt_train.py  --depth=3  &
# python3 sdf_train.py --num_trees=3 --depth=3  & 
python3 sdf_module_train.py --num_trees=9 --depth=5

min=3
max=9
inter=2

#for ((i=min; i <= max; i+=inter));
#do 
#    python3 sdf_module_train.py --num_trees="$i" --depth=5 &
#done 
