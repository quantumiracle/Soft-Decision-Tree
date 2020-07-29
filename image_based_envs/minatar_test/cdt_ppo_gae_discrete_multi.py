import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from minatar import Environment
import matplotlib.pyplot as plt
import numpy as np
from cascade_tree import Cascade_DDT 
from gym_wrapper import GymWrapper, LowDimWrapper
import torch.multiprocessing as mp
from torch.multiprocessing import Process

torch.multiprocessing.set_start_method(
    'forkserver', force=True)  # critical for make multiprocessing work

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
Episodes      = 100000
T_horizon     = 1000
NUM_WORKERS = 1
EnvName = 'freeway'

dSiLU = lambda x: torch.sigmoid(x)*(1+x*(1-torch.sigmoid(x)))
SiLU = lambda x: x*torch.sigmoid(x)

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, learner_args):
        super(PPO, self).__init__()
        self.data = []
        self.model_path = learner_args['model_path']
        self.device = learner_args['device']
        hidden_dim=128
        self.fc1   = nn.Linear(state_dim,hidden_dim)
        self.fc2   = nn.Linear(hidden_dim,hidden_dim)
        # self.fc_pi = nn.Linear(hidden_dim,action_dim)
        self.fc_v  = nn.Linear(hidden_dim,1)

        self.cdt = Cascade_DDT(learner_args).to(self.device)
        # self.pi = lambda x: self.cdt.forward(x, LogProb=False)[1]

        self.optimizer = optim.Adam(list(self.parameters())+list(self.cdt.parameters()), lr=learning_rate)

    def _shared_body(self,x):
        x=SiLU(self.fc1(x))
        x=dSiLU(self.fc2(x))
        return x

    def pi(self, x):
        return self.cdt.forward(x, LogProb=False)[1]   
    
    def v(self, x):
        if isinstance(x, (np.ndarray, np.generic) ):
            x = torch.tensor(x)
        x = self._shared_body(x)
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
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                          torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach()

            advantage_lst = []
            advantage = 0.0
            for delta_t in torch.flip(delta, [0]):
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.pi(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def choose_action(self, s):
        prob = self.pi(torch.from_numpy(s).unsqueeze(0).float().to(self.device)).squeeze()
        m = Categorical(prob)
        a = m.sample().item()
        return a, prob

    def save_model(self, path=None):
        torch.save(self.state_dict(), path + '_policy')

    def load_model(self, ):
        self.load_state_dict(torch.load(self.model_path))

def ShareParameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()

def run(id, EnvName, model, learner_args, rewards_queue, train=False, test=False):
    env = LowDimWrapper(Environment(EnvName))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    print(state_dim, action_dim)
    print_interval = 20
    for n_epi in range(Episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            for t in range(T_horizon):
                a, prob = model.choose_action(s)
                s_prime, r, done, info = env.step(a)
                if test:
                    env.render()
                # model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                model.put_data((s, a, float(r), s_prime, prob[a].item(), done))

                s = s_prime

                reward += r
                step+=1
                if done:
                    break
            if train:
                model.train_net()
        if train:   
            if rewards_queue is not None:
                rewards_queue.put(reward)
            if n_epi%print_interval==0 and n_epi!=0:
                torch.save(model.state_dict(), learner_args['model_path'])
                print("worker ID :{}, # of episode :{}, reward : {:.1f}, episode length: {}".format(id, n_epi, reward, step))
        else:
            print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))

    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--depth1', dest='feature_learning_depth', default=False)
    parser.add_argument('--depth2', dest='decision_depth', default=False)
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--id', dest='id', default=False)

    args = parser.parse_args()

    env = LowDimWrapper(Environment(EnvName))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    env.close()

    learner_args = {
        'num_intermediate_variables': 10,
        'feature_learning_depth': int(args.feature_learning_depth),
        'decision_depth': int(args.decision_depth),
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

    path='cdt_ppo_discrete_'+EnvName+'depth_'+args.feature_learning_depth+args.decision_depth+'_id'+str(args.id)
    learner_args['path'] = path
    learner_args['model_path'] = './model_cdt_ppo/'+path
    learner_args['device'] = torch.device('cuda' if learner_args['cuda'] else 'cpu')

    model = PPO(state_dim, action_dim, learner_args,).to(learner_args['device'])

    if args.train:
        model.share_memory()
        ShareParameters(model.optimizer)
        rewards_queue = mp.Queue(
        )  # used for get rewards from all processes and plot the curve
        processes = []
        rewards = []
        for i in range(NUM_WORKERS):
            process = Process(target=run,
                              args=(
                                  i,
                                  EnvName,
                                  model,
                                  learner_args,
                                  rewards_queue,
                                  True,
                                  False
                              ))  # the args contain shared and not shared
            process.daemon = True  # all processes closed when the main stops
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

            if len(rewards) % 20 == 0 and len(rewards) > 0:
                np.save('./log/'+learner_args['path'], rewards)

        [p.join() for p in processes]  # finished at the same time

        model.save_model(learner_args['model_path'])

    if args.test:
        model.load_model()
        run(0, EnvName, model, learner_args, rewards_queue=None, train=False, test=True)
