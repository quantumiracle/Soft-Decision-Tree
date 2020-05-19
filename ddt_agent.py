# -*- coding: utf-8 -*-
'''' 
Agent from paper DDT
'''
import torch
import torch.nn as nn
import numpy as np
import gym
from heuristic_tree_agent2 import run


def DDT_action(s):
    if s[5]>0.04:
        a=3
    elif s[3]>0.05:
        a=1
    elif s[3]>0.26:
        a=0
    elif s[3]>-0.35:
        a=0
    elif s[3]>-0.04:
        a=0
    elif s[5]>-0.22:
        a=0
    elif s[4]>-0.04:
        a=1
    elif s[6]>0:
        a=2
    elif s[2]>0.32:
        a=2
    elif s[6]>0:
        a=3
    elif s[5]>0.16:
        a=2
    elif s[3]>0.19:
        a=3
    elif s[3]>0.4:
        a=0
    elif s[1]>-0.11:
        a=1
    elif s[4]>0.15:
        a=1
    elif s[0]>-0.34:
        a=1
    else:
        a=3
    return a

if __name__ == '__main__':
    model = lambda x: DDT_action(x)
    run(model, episodes=100)
