import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from minatar import Environment
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from env_wrapper import DiscreteActionWrapper, ObservationWrapper
from common.utils import * 

import math

torch.multiprocessing.set_start_method('forkserver', force=True) # critical for make multiprocessing work

# EnvName = 'CarRacing-v0'
# EnvName = 'Enduro-v0'
#EnvName = 'Freeway-v0'
EnvName = 'MsPacman-v0'


#Hyperparameters
learning_rate = 0.0005
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 1000
TRAIN_EPI     = 20000
NUM_WORKERS   = 1
MODEL_PATH = './model/single_ppo_discrete_'+EnvName

class PPO(nn.Module):
    def __init__(self, obs_space, action_space, hidden_dim=128):
        super(PPO, self).__init__()
        self.data = []
        self.obs_space = obs_space
        self.action_space = action_space
        self.action_dim = action_space.n  # discrete
        if len(obs_space.shape) == 1:
            self.in_layer   = nn.Linear(obs_space.shape[0],hidden_dim)
            in_layer_dim = hidden_dim
        else: # high-dimensional inputs
            X_channel = self.obs_space.shape[0]
            X_dim1 = self.obs_space.shape[1]
            X_dim2 = self.obs_space.shape[2]
            print(self.obs_space)
            # assert self.obs_space.shape[1] == self.obs_space.shape[2]
            self.CONV_NUM_FEATURE_MAP=12
            self.CONV_KERNEL_SIZE=4
            self.CONV_STRIDE=1
            self.CONV_PADDING=0
            self.in_layer1 = nn.Sequential(
                nn.Conv2d(X_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
                nn.ReLU())
            self.in_layer2 = nn.Sequential(
                nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
                # nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
                nn.ReLU(),
            )
            dim1_conv_size1 = int((X_dim1-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
            dim1_conv_size2 = int((dim1_conv_size1-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
            dim2_conv_size1 = int((X_dim2-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
            dim2_conv_size2 = int((dim2_conv_size1-self.CONV_KERNEL_SIZE+2*self.CONV_PADDING)/self.CONV_STRIDE) + 1
            in_layer_dim = int(self.CONV_NUM_FEATURE_MAP*2* dim1_conv_size2*dim2_conv_size2)
        self.fc_h1 = nn.Linear(in_layer_dim, hidden_dim)
        self.fc_h2 = nn.Linear(in_layer_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim,self.action_dim)  
        self.fc_v  = nn.Linear(hidden_dim,1)
        self.optimizer = SharedAdam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = -1):
        if len(x.shape) >1:
            if len(x.shape) ==3:
                x = x.unsqueeze(0)
            x = self.in_layer1(x)
            x = self.in_layer2(x)
            x = x.view(x.shape[0], -1)
        else:
            x = self.in_layer1(x)
            x = self.in_layer2(x)
        x = dSiLU(self.fc_h1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        if len(x.shape) >1:
            if len(x.shape) ==3:
                x = x.unsqueeze(0)
            x = self.in_layer1(x)
            x = self.in_layer2(x)            
            x = x.view(x.shape[0], -1)
        else:
            x = self.in_layer1(x)
            x = self.in_layer2(x)
        x = dSiLU(self.fc_h2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
                                          torch.tensor(r_lst).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda(), \
                                          torch.tensor(done_lst, dtype=torch.float).cuda(), torch.tensor(prob_a_lst).cuda()
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        r = r.float()
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def choose_action(self, s):
        # reshape the input
        s= torch.from_numpy(s).float().cuda()
        prob = self.pi(s).squeeze()
        m = Categorical(prob)
        a = m.sample().item()
        return a, prob

    def save_model(self, path=MODEL_PATH):
        torch.save(self.state_dict(), path+'_ac')

    def load_model(self, path=MODEL_PATH):
        self.load_state_dict(torch.load(MODEL_PATH, map_location='cuda:0'))
        self.eval()
    
def run(id, model, rewards_queue, train=False, test=False):
    with torch.cuda.device(id % torch.cuda.device_count()):
        model.cuda()
        # env = DiscreteActionWrapper(gym.make(EnvName))
        env = ObservationWrapper(gym.make(EnvName), selected_channel='grey')

        episode_r = 0.0
        print_interval = 10        
        Epi_r = []
        Epi_length = []
        for n_epi in range(TRAIN_EPI):
            s = env.reset()
            episode_r = 0.0
            step=0
            done = False
            while not done:
                a, prob = model.choose_action(s)

                s_prime, r, done, info = env.step(a)
                if test:
                    env.render()
                # model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                model.put_data((s, a, r, s_prime, prob[a].item(), done))

                s = s_prime

                episode_r += r
                step+=1
                if done:
                    break
            if train:
                model.train_net()
            if rewards_queue is not None:
                rewards_queue.put(episode_r)
            Epi_r.append(episode_r)
            Epi_length.append(step)
            if n_epi%print_interval==0 and n_epi!=0:
                if train:
                    torch.save(model.state_dict(), MODEL_PATH)
            print("Worker ID: {} | Episode :{} | Average episode reward : {:.3f} | Average episode length: {}".format(id, n_epi, np.mean(Epi_r), np.mean(Epi_length)))
            Epi_length = []
            Epi_r = []
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    args = parser.parse_args()
    
    # env = DiscreteActionWrapper(gym.make(EnvName))
    env = ObservationWrapper(gym.make(EnvName), selected_channel='grey')

    model = PPO(env.observation_space, env.action_space)

    if args.train:
        model.share_memory()
        ShareParameters(model.optimizer)
        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        processes=[]
        rewards=[]
        for i in range(NUM_WORKERS):
            process = Process(target=run, args=(i, model, rewards_queue, True, False))  # the args contain shared and not shared
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)
        [p.start() for p in processes]
        while True:  # keep geting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                if len(rewards) == 0:
                    rewards.append(r)
                else:
                    rewards.append(rewards[-1] * 0.9 + r * 0.1)
            else:
                break

            if len(rewards)%20==0 and len(rewards)>0:
                np.save('single_ppo_learn_'+EnvName, rewards)

        [p.join() for p in processes]  # finished at the same time

        model.save_model(MODEL_PATH)
        # run(env, train=True, test=False)
    if args.test:
        model.load_model(MODEL_PATH)
        run(0, model, rewards_queue=None, train=False, test=True)
