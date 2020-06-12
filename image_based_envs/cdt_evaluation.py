# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from cdt_plot import draw_tree, get_path
import sys
sys.path.insert(0,'..')
from heuristic_evaluation import normalize
from env_wrapper import ObservationWrapper
import os

# EnvName = 'CartPole-v1'  # LunarLander-v2
# EnvName = 'LunarLander-v2' 
EnvName = 'Freeway-v0'

def evaluate(model, tree, episodes=1, frameskip=1, seed=None, DrawTree=None, DrawImportance=True, img_path = 'img/eval_tree'):
    env = ObservationWrapper(gym.make(EnvName), selected_channel='grey')
    if seed:
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    average_weight_list = []

    # # show values on tree nodes
    # print(tree.state_dict())
    # # show probs on tree leaves
    # softmax = nn.Softmax(dim=-1)
    # print(softmax(tree.state_dict()['dc_leaves']).detach().cpu().numpy())

    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        average_weight_list_epi = []
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            # a = model(s)
            a=1
            print(a)
            if a==0:
                if step%frameskip==0:
                    if DrawTree is not None:
                        draw_tree(tree, input_img=np.moveaxis(s, 0, 2), DrawTree=DrawTree, savepath=img_path+'_'+DrawTree+'/{:04}.png'.format(step))

            s_prime, r, done, info = env.step(a)
            env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
    # np.save('data/s/dt_importance.npy', average_weight_list)

    env.close()




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
    #Freeway PPO
    from mp_cdt_ppo_gae_discrete_cnn import PPO, MODEL_PATH
    
    # for reproduciblility
    seed=3
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = ObservationWrapper(gym.make(EnvName), selected_channel='grey')

    if len(env.observation_space.shape)>1:
        state_dim=1
        for dim in env.observation_space.shape:
            state_dim*=dim
    else:
        state_dim=env.observation_space.shape[0]
    action_dim = env.action_space.n

    learner_args = {
    'num_intermediate_variables': 50,
    'feature_learning_depth': 2,
    'decision_depth': 2,
    'input_dim': state_dim,
    'output_dim': action_dim,
    'lr': 1e-3,
    'weight_decay': 0.,  # 5e-4
    'batch_size': 1280,
    'exp_scheduler_gamma': 1.,
    'cuda': True,
    'epochs': 40,
    'log_interval': 100,
    'greatest_path_probability': True,
    'beta_fl' : False,  # temperature for feature learning
    'beta_dc' : False,  # temperature for decision making
    }

    ppo = PPO(env.observation_space, env.action_space, learner_args)
    ppo.load_model(MODEL_PATH)
    tree = ppo.cdt

    num_params = 0
    for key, v in tree.state_dict().items():
        print(key, v.shape)
        num_params+=v.reshape(-1).shape[0]
    print('Total number of parameters in model: ', num_params)

    model = lambda x: ppo.choose_action(x, Greedy=True)
    img_path = 'img/eval_tree_{}'.format(tree.args['num_intermediate_variables'])

    evaluate(model, tree, episodes=10, frameskip=1, seed=seed, DrawTree='FL', DrawImportance=False, img_path=img_path)

