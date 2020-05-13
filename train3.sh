#!/bin/bash

# hostname
echo $CUDA_VISIBLE_DEVICES

python3 sdf_module_train.py --num_trees=3 --depth=3 
