# -*- coding: utf-8 -*-
'''' 
Soft Decision Forests: 
ensemble of Soft Decision Trees
'''
import torch
import torch.nn as nn
import numpy as np
import gym

node_list = [
{6:1, 7:1},
{0:0.25, 2:0.5, 4:-0.5, 5:-1},
{3:-0.5},
{3:-0.5, 8:-0.05},
{0:0.25, 2:0.5, 4:-0.5, 5:-1},
{0:0.025, 1:-0.5, 2:-0.5, 3:-0.5, 4:0.5, 5:1},
{0:0.525, 1:-0.5, 2:0.5, 3:-0.5, 4:-0.5, 5:-1},
{0:0.275, 1:-0.5, 3:-0.5, 8:-0.05},
{0:0.25, 2:0.5, 4:-0.5, 5:-1, 8:-0.05},
{0:0.275, 1:-0.5, 3:-0.5, 8:-0.05},
{0:-0.25, 2:-0.5, 4:0.5, 5:1, 8:-0.05},
{0:0.25, 2:0.5, 4:-0.5, 5:-1, 8:-0.05},
{0:-0.25, 2:-0.5, 4:0.5, 5:1, 8:-0.05},
{0:0.25, 2:0.5, 4:-0.5, 5:-1},
{0:-0.525, 1:-0.5, 2:-0.5, 3:-0.5, 4:0.5, 5:1},
{0:-0.025, 1:-0.5, 2:0.5, 3:-0.5, 4:-0.5, 5:-1},
{0:-0.275, 1:-0.5, 3:-0.5, 8:-0.05},
{0:-0.275, 1:-0.5, 3:-0.5, 8:-0.05},
2,
1,
0,
1,
0,
2,
3,
0,
3,
0
]

child_list = [
    [2, 1],
    [4, 13],
    [3, 27],
    [23, 25],
    [5, 6],
    [7, 8],
    [9, 10],
    [18, 11],
    [21, 22],
    [23, 12],
    [26, 27],
    [19, 20],
    [24, 25],
    [14, 15],
    [16, 8],
    [17, 10],
    [18, 11],
    [23, 12]
]



class Node(object):
    def __init__(self, id, weights, left_child_id, right_child_id):
        super(Node, self).__init__()
        self.id = id
        self.weights = weights
        self.left_child_id = left_child_id
        self.right_child_id = right_child_id

    def decide(self, aug_x):
        prod = np.sum(self.weights*aug_x)  # weights include bias
        if prod>0:
            return self.left_child_id
        else:
            return self.right_child_id


class Leaf(object):
    def __init__(self, id, value):
        super(Leaf, self).__init__()
        self.id = id
        self.value = value

def dict_to_vector(dict, dim=8):
    v = np.zeros(dim+1)  # the last dim is bias
    for key, value in dict.items():
        v[key]=value
    return v


class HeuristicTree(object):
    def __init__(self, node_list, child_list):
        super(HeuristicTree, self).__init__()
        self.node_list=[]
        for i, node in enumerate(node_list):
            if isinstance(node, dict):  # inner node
                w = dict_to_vector(node)
                self.node_list.append(Node(i, w, child_list[i][0], child_list[i][1]))
            else: # leaf
                self.node_list.append(Leaf(i, node))

    def forward(self, x, Path=True):
        aug_x = np.concatenate((np.array(x), [1]))
        child = self.node_list[0].decide(aug_x)
        decision_path=[0]
        while isinstance(self.node_list[child], Node):
            child = self.node_list[child].decide(aug_x)
            decision_path.append(child)
        if Path:
            return self.node_list[child].value, decision_path
        else:
            return self.node_list[child].value


def run(model, episodes=1, seed=None):
    env = gym.make('LunarLander-v2')
    if seed:
        env.seed(seed)
    for n_epi in range(episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(s)
            s_prime, r, done, info = env.step(a)
            env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))


if __name__ == '__main__':  
    tree = HeuristicTree(node_list, child_list)
    model = lambda x: tree.forward(x, Path=False)
    run(model, episodes=100)