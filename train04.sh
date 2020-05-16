#!/bin/bash

hostname
echo $CUDA_VISIBLE_DEVICES

min=4
max=6
inter=1
for ((i=min; i <= max; i+=inter));
do 
    python3 sdt_train.py  --depth=3 --id="$i"
    #python3 sdf_train.py --num_trees=3 --depth=9 --id="$i"
    #python3 sdf_module_train.py --num_trees=7 --depth=5 --id="$i"
done
