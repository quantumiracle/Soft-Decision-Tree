# -*- coding: utf-8 -*-
'''' 
A decision tree of heuristic agent for LunarLander
'''
import torch
import torch.nn as nn
import numpy as np
import gym

node_list = [
{6:1, 7:1},
{0:0.5, 2:1, 8:-0.4},  # last dim is bias
{0:-0.5, 2:-1, 8:-0.4},
{0:1},
{0:1},
{0:1},
[9*[0], [0,0,0,-0.5, 0,0,0,0,0]],  # [at, ht]
[[0,0,0,0,-0.5,-1,0,0,0.2], [0.275, -0.5, 0,-0.5,0,0,0,0,0]], 
[[0,0,0,0,-0.5,-1,0,0,0.2], [-0.275, -0.5, 0,-0.5,0,0,0,0,0]],
[[0,0,0,0,-0.5,-1,0,0,-0.2], [0.275, -0.5, 0,-0.5,0,0,0,0,0]], 
[[0,0,0,0,-0.5,-1,0,0,-0.2], [-0.275, -0.5, 0,-0.5,0,0,0,0,0]],
[[0.25, 0,0.5,0,-0.5, -1,0,0,0], [0.275, -0.5, 0,-0.5,0,0,0,0,0]], 
[[0.25, 0,0.5,0,-0.5, -1,0,0,0], [-0.275, -0.5, 0,-0.5,0,0,0,0,0]],
]

child_list = [
    [6,1],
    [3,2], 
    [4,5],
    [7,8],
    [9,10],
    [11,12]
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
            return [self.left_child_id]
        else:
            return [self.right_child_id]


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
                w = dict_to_vector(node, dim=8)
                self.node_list.append(Node(i, w, child_list[i][0], child_list[i][1]))
            else: # leaf
                self.node_list.append(HeuristicTreeSub1(sub1_node_list, sub1_child_list, node[0], node[1], i))

    def forward(self, x, Info=True):
        aug_x = np.concatenate((np.array(x), [1]))
        child = self.node_list[0].decide(aug_x)[0]
        decision_path=[self.node_list[0].id]
        decision_weights=[self.node_list[0].weights]
        while True:
            last_child = child
            info = self.node_list[child].decide(aug_x)
            child=info[0]
            if isinstance(self.node_list[last_child], HeuristicTreeSub1):
                break
            decision_path.append(last_child)
            decision_weights.append(self.node_list[last_child].weights)

        sub_tree_path = info[1]
        sub_tree_weights = info[2]
        decision_path+=sub_tree_path
        decision_weights+=sub_tree_weights

        if Info:
            return [child, decision_path, decision_weights]
        
        else:
            return child


sub1_node_list=[
    {0:1},  # 0: at, 1: ht
    {0:-1, 1:1}, 
    {0:1, 1:1}, 
    {1:1, 2:-0.05},
    {0:1, 2:-0.05}, 
    {1:1, 2:-0.05}, 
    {0:-1, 2:-0.05}, 
    {0:1, 2:-0.05}, 
    {0:-1, 2:-0.05},
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

sub1_child_list=[
    [1,2],
    [3,4],
    [5,6],
    [9,7],
    [12,13],
    [14,8],
    [17,18],
    [10,11],
    [15,16]
]


class SubNode(object):
    def __init__(self, id, weights, at, ht, left_child_id, right_child_id):
        super(SubNode, self).__init__()
        self.id = id
        self.weights = weights[0]*np.array(at)+weights[1]*np.array(ht)
        self.weights[-1]+=weights[-1]
        self.left_child_id = left_child_id
        self.right_child_id = right_child_id

    def decide(self, aug_x):
        prod = np.sum(self.weights*aug_x)  # weights include bias
        if prod>0:
            return self.left_child_id
        else:
            return self.right_child_id


class HeuristicTreeSub1(object):
    def __init__(self, node_list, child_list, at, ht, tree_id):
        super(HeuristicTreeSub1, self).__init__()
        self.node_list=[]
        for i, node in enumerate(node_list):
            sub_id = 'sub_'+str(i)  # idex of node on subtree 
            if isinstance(node, dict):  # inner node
                w = dict_to_vector(node, dim=2)
                self.node_list.append(SubNode(sub_id, w, at, ht, child_list[i][0], child_list[i][1]))
            else: # leaf
                self.node_list.append(Leaf(sub_id, node))

    def decide(self, aug_x, Path=False):
        child = self.node_list[0].decide(aug_x)
        decision_path=[self.node_list[0].id]
        weights_list=[self.node_list[0].weights]
        while isinstance(self.node_list[child], SubNode):
            weights_list.append(self.node_list[child].weights)
            decision_path.append(self.node_list[child].id)
            child = self.node_list[child].decide(aug_x)
        decision_path.append(self.node_list[child].id)  # add leaf
        return [self.node_list[child].value, decision_path, weights_list]


def run(model, episodes=1, seed=None):
    env = gym.make('LunarLander-v2')
    import time
    if seed:
        env.seed(seed)
    for n_epi in range(episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(s)
            print(a)
            s_prime, r, done, info = env.step(a)
            env.render()
            s = s_prime
            time.sleep(0.1)
            reward += r
            step+=1
            if done:
                break

        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))


def evaluate(model, episodes=1, frameskip=1, seed=None):
    from heuristic_evaluation import normalize
    from sdt_evaluation import plot_importance_single_episode

    env = gym.make('LunarLander-v2')
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    average_weight_list = []

    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        average_weight_list_epi = []
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            info = model(s)
            a=info[0]
            if step%frameskip==0:
                average_weight = np.mean(np.abs(normalize(np.array(info[2])[:, :-1])), axis=0) # take absolute to prevent that positive and negative will counteract
                average_weight_list_epi.append(average_weight)

            s_prime, r, done, _ = env.step(a)
            # env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
    path = 'data/heuristic_tree_importance_offline.npy'
    np.save(path, average_weight_list)
    plot_importance_single_episode(data_path=path, save_path='./img/heuristic_tree_importance_online.png', epi_id=0)


    env.close()


def evaluate_offline(model, episodes=1, frameskip=1, seed=None, data_path='./data/evaluate_state.npy', DrawImportance=True, method='weight'):
    from heuristic_evaluation import normalize
    from sdt_evaluation import plot_importance_single_episode

    states = np.load(data_path, allow_pickle=True)
    average_weight_list=[]
    for n_epi in range(episodes):
        average_weight_list_epi = []
        for i, s in enumerate(states[n_epi]):
            info = model(s)
            if i%frameskip==0:
                if DrawImportance:
                    if method == 'weight': 
                        average_weight = np.mean(np.abs(normalize(np.array(info[2])[:, :-1])), axis=0) # take absolute to prevent that positive and negative will counteract
                        average_weight_list_epi.append(average_weight)
                    elif method == 'gradient':  # not finished yet, need to write the heuristic decision tree with torch tensor rather than np
                        x = torch.Tensor([s])
                        x.requires_grad = True
                        a = model(x)[0] # [1] is output, which requires gradient, but it's the expectation of leaves rather than the max-prob leaf 
                        gradient = torch.autograd.grad(outputs=a, inputs=x, grad_outputs=torch.ones_like(a),
                                            retain_graph=True, allow_unused=True)
                        # print('grad:', gradient[0].squeeze())
                        average_weight_list_epi.append(np.abs(gradient[0].squeeze().cpu().numpy()))

        average_weight_list.append(average_weight_list_epi)
    path = 'data/heuristic_tree_importance_offline.npy'
    np.save(path, average_weight_list)
    plot_importance_single_episode(data_path=path, save_path='./img/heuristic_tree_importance_offline.png', epi_id=0)


if __name__ == '__main__':  
    tree = HeuristicTree(node_list, child_list)
    # RL test
    # model = lambda x: tree.forward(x, Info=False)
    # run(model, episodes=100)

    # single instance with full information
    # x=np.zeros(8)
    # print(tree.forward(x, Info=True))

    # tree evaluation
    model = lambda x: tree.forward(x, Info=True)
    evaluate(model, episodes=1, seed=3)
    # evaluate_offline(model, episodes=1, seed=3)
    from sdt_evaluation import plot_importance_single_episode
    plot_importance_single_episode(data_path='data/heuristic_tree_importance.npy', save_path='./img/heuristic_tree_importance.png',)