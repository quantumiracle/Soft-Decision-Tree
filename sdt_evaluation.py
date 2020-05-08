# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tree_plot import draw_tree
import os


def run(model, tree, episodes=1, frameskip=1, seed=0):
    env = gym.make('LunarLander-v2')
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    for n_epi in range(episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(torch.Tensor([s]))
            if step%frameskip==0:
                img_path='img/eval_abs2_{}'.format(tree.args['depth'])
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                draw_tree(tree, (tree.args['input_dim'],), input_img=s, savepath=img_path+'/{:04}.png'.format(step))

            s_prime, r, done, info = env.step(a)
            env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))

    env.close()

if __name__ == '__main__':
    from sdt_train import learner_args
    from SDT import SDT

    # for reproduciblility
    seed=3
    torch.manual_seed(seed)
    np.random.seed(seed)

    learner_args['depth']=3
    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])

    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().cpu().numpy()
    run(model, tree, episodes=1, frameskip=1, seed=seed)