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


def run(model, tree, episodes=1, frameskip=1, seed=0, SaveImg=True, DrawImportance=True):
    env = gym.make('LunarLander-v2')
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    img_path='img/eval_abs2_{}'.format(tree.args['depth'])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    tree_weights = tree.get_tree_weights()
    average_weight_list = []

    for n_epi in range(episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(torch.Tensor([s]))
            if step%frameskip==0 and SaveImg:
                path_idx = draw_tree(tree, (tree.args['input_dim'],), input_img=s, savepath=img_path+'/{:04}.png'.format(step))
                if DrawImportance:
                    weights_on_path = tree_weights[path_idx[:-1]]  # remove leaf node, i.e. the last index 
                    average_weight = np.mean(np.abs(weights_on_path), axis=0)
                    average_weight_list.append(average_weight)

            s_prime, r, done, info = env.step(a)
            env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        np.save('data/sdt_importance.npy', average_weight_list)

        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))

    env.close()


def plot_importance(data_path='data/sdt_importance.npy', save_path=None):
    data = np.load(data_path)
    for i, weights_per_feature in enumerate(data.T):
        plt.plot(weights_per_feature, label='Dim: {}'.format(i))
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    from sdt_train import learner_args
    from SDT import SDT

    # for reproduciblility
    seed=15
    torch.manual_seed(seed)
    np.random.seed(seed)

    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])

    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().cpu().numpy()
    run(model, tree, episodes=1, frameskip=1, seed=seed)
    # plot_importance()