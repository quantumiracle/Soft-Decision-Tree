# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tree_plot import draw_tree, get_path
from heuristic_evaluation import normalize
import os

def evaluate(model, tree, episodes=1, frameskip=1, seed=None, DrawTree=True, DrawImportance=True):
    env = gym.make('LunarLander-v2')
    if seed:
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    img_path='img/eval_abs2_{}'.format(tree.args['depth'])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    tree_weights = tree.get_tree_weights()
    average_weight_list = []

    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        average_weight_list_epi = []
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(torch.Tensor([s]))
            if step%frameskip==0:
                if DrawTree:
                    draw_tree(tree, (tree.args['input_dim'],), input_img=s, savepath=img_path+'/{:04}.png'.format(step))
                if DrawImportance:
                    path_idx = get_path(tree, s)
                    weights_on_path = tree_weights[path_idx[:-1]]  # remove leaf node, i.e. the last index 
                    average_weight = np.mean(np.abs(normalize(weights_on_path)), axis=0)  # take absolute to prevent that positive and negative will counteract
                    average_weight_list_epi.append(average_weight)

            s_prime, r, done, info = env.step(a)
            # env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
    path = 'data/sdt_importance_online.npy'
    np.save(path, average_weight_list)
    plot_importance_single_episode(data_path=path, save_path='./img/sdt_importance_online.png', epi_id=0)

    env.close()

def evaluate_offline(model, tree, episodes=1, frameskip=1, seed=None, data_path='./data/evaluate_state.npy', DrawImportance=True, method='weight'):
    states = np.load(data_path, allow_pickle=True)
    tree_weights = tree.get_tree_weights()
    average_weight_list=[]
    for n_epi in range(episodes):
        average_weight_list_epi = []
        for i, s in enumerate(states[n_epi]):
            a = model(torch.Tensor([s]))    
            if i%frameskip==0:
                if DrawImportance:
                    if method == 'weight': 
                        path_idx = get_path(tree, s)
                        weights_on_path = tree_weights[path_idx[:-1]]  # remove leaf node, i.e. the last index 
                        average_weight = np.mean(np.abs(normalize(weights_on_path)), axis=0)  # take absolute to prevent that positive and negative will counteract
                        average_weight_list_epi.append(average_weight)
                    elif method == 'gradient':
                        x = torch.Tensor([s])
                        x.requires_grad = True
                        a = tree.forward(x)[1] # [1] is output, which requires gradient, but it's the expectation of leaves rather than the max-prob leaf 
                        gradient = torch.autograd.grad(outputs=a, inputs=x, grad_outputs=torch.ones_like(a),
                                            retain_graph=True, allow_unused=True)
                        # print('grad:', gradient[0].squeeze())
                        average_weight_list_epi.append(np.abs(gradient[0].squeeze().cpu().numpy()))
                        # average_weight_list_epi.append(gradient[0].squeeze().cpu().numpy())


        average_weight_list.append(average_weight_list_epi)
    path = 'data/sdt_importance_offline.npy'
    np.save(path, average_weight_list)
    plot_importance_single_episode(data_path=path, save_path='./img/sdt_importance_offline.png', epi_id=0)


def collect_offline_states(episodes=100):
    """Collect episodes of states data from heuristic agent"""
    from lunar_lander_heuristic import Heuristic_agent
    Continuous = False
    if Continuous:
        env = gym.make('LunarLanderContinuous-v2').unwrapped
    else:
        env = gym.make('LunarLander-v2').unwrapped
    heuristic_agent = Heuristic_agent(env, Continuous)

    if seed:
        env.seed(seed)
    s_data=[]
    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        s=env.reset()
        s_list=[]
        done=False
        while not done:
            # env.render()
            s_list.append(s)
            a=heuristic_agent.choose_action(s)
            s, r, done, info = env.step(a)
            if done: break
        s_data.append(s_list)
    np.save('./data/evaluate_state', s_data)


def plot_importance_single_episode(data_path='data/sdt_importance.npy', save_path='./img/sdt_importance.png', epi_id=0):
    data = np.load(data_path, allow_pickle=True)[epi_id]
    markers=[".", "d", "o", "*", "^", "v", "p", "h"]
    for i, weights_per_feature in enumerate(np.array(data).T):
        plt.plot(weights_per_feature, label='Dim: {}'.format(i), marker=markers[i], markevery=8)
    plt.legend(loc=1)
    plt.xlabel('Step')
    plt.ylabel('Feature Importance')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    from sdt_train import learner_args
    from SDT import SDT

    # for reproduciblility
    seed=3
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    learner_args['cuda'] = False  # cpu
    learner_args['beta'] = True
    learner_args['lamda'] = 0.001
    learner_args['model_path']='./model/trees/sdt_'+str(learner_args['lamda'])+'_id'+str(1)
    # learner_args['model_path']='./model/trees/sdt_'+str(learner_args['lamda'])+'_id'+str(1)+'beta'
    # learner_args['model_path']='./model/trees/sdt_'+str(learner_args['lamda'])+'_id'+str(1)+'beta'+'_discretized'

    tree = SDT(learner_args)
    tree.load_model(learner_args['model_path'])
    # collect_offline_states()
    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().numpy()

    # evaluate(model, tree, episodes=1, frameskip=1, seed=seed, DrawTree=False, DrawImportance=False)
    evaluate_offline(model, tree, episodes=1, frameskip=1, seed=seed, DrawImportance=True, method='gradient')

    # plot_importance_single_episode(epi_id=0)