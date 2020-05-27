import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from torch.distributions import Categorical
import torch
from sklearn import tree
from joblib import dump, load


# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.


class Heuristic_agent():

    def __init__(self, env, Continuous):
        super(Heuristic_agent, self).__init__()
        self.continuous = Continuous
    
    def choose_action(self, s,  DIST = False):
        # Heuristic for:
        # 1. Testing. 
        # 2. Demonstration rollout.
        angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4: angle_targ = -0.4
        hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]: # legs have contact
            angle_todo = 0
            hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

        if self.continuous:
            a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
            a = np.clip(a, -1, +1)
        else:
            a = 0  # do nothing
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2  # fire main
            elif angle_todo < -0.05: a = 3  # fire right
            elif angle_todo > +0.05: a = 1  # fire left
        return a

class Simple_heuristic_agent():
    def __init__(self, env, Continuous):
        super(Simple_heuristic_agent, self).__init__()
        self.continuous = Continuous
    
    def choose_action(self, s, DIST = False):
        offset_x = abs(s[0])
        abs_y = abs(s[1])
        if self.continuous:
            # if goes too far from center of X, fire conter side to tune
            x_threshold = 0.05  # the smaller the threshold, the more sensitive to this condition
            if s[0]>x_threshold:
                angle_todo = -1.1*(s[0]-x_threshold)
            elif s[0]<-x_threshold:
                angle_todo = - 1.1*(s[0] - (-x_threshold))
            else:
                angle_todo = 0.
            
            # if angle is too large, fire side to tune
            angle_threshold = 0.1  # the smaller the threshold, the more sensitive to this condition
            if s[4]>angle_threshold:
                angle_todo += s[4] - angle_threshold
            elif s[4]< -angle_threshold:
                angle_todo += s[4]- (-angle_threshold)

            # main fire in proper range and inverse proportional to the height
            if s[1]<0.9 and s[1]>0.15:
                hover_todo = 1.4-s[1]
            else:
                hover_todo =0.
            
            a = np.array([hover_todo , angle_todo*5] )
            a = np.clip(a, -1, +1)

        else:  # discrete env
            a=0
            # if contact or close to ground, fire main to slow down
            # if s[6] or s[7] or (offset_x >0.1 and offset_x< 0.2) or (s[1]<0.8 and s[1]>0.15):  # y-axis starting position around 1.4
            if s[6] or s[7] or (s[1]<0.8 and s[1]>0.15):  # y-axis starting position around 1.4            
                a = 2

            # if goes too far from center of X, fire counter side to tune
            x_threshold = 0.4 # the smaller the threshold, the more sensitive to this condition
            if s[0]<-x_threshold:
                a=3
            elif s[0]>x_threshold:
                a=1

            # if angle is too large, fire side to tune;
            # this is one lever higher than the logic above, to prevent over rotation
            angle_threshold = 0.3  # the smaller the threshold, the more sensitive to this condition
            if s[4]>angle_threshold:
                a=3
            elif s[4]<-angle_threshold:
                a=1
            
        print('offset x: ', offset_x, s[4])

        return a, None

def select_action_from_node(value_list):
    # normalize
    total = np.sum(value_list)
    if total>0:
        prob = np.array(value_list)/float(total)
    dist = Categorical(torch.from_numpy(prob))
    a = dist.sample().item()
    return a, dist



class Decision_tree_agent():  # discrete; not work well
    def __init__(self, env, Continuous, depth=None):
        super(Decision_tree_agent, self).__init__()
        self.continuous = Continuous
        self.depth = depth
        self.model_prefix = './model/continuous_' if self.continuous else './model/discrete_'
        self.data_prefix = './data/continuous_' if self.continuous else './data/discrete_'
        self.load_split_data(train_test_ratio=0.8)
        try:  # if tree model exists, load it; else, fit one.
            self.tree = load(self.model_prefix+'decision_tree{}.joblib'.format(self.depth))
            print('Tree loaded! Depth: {}  Accuracy: {} '.format(self.depth, self.evaluate_accuracy(self.tree)))
        except:
            self.fit_model()

    def load_split_data(self, train_test_ratio):
        s = np.load(self.data_prefix+'state.npy')
        a = np.load(self.data_prefix+'action.npy')
        a = a.reshape(a.shape[0], -1)
        # split train test data
        s_split_idx = int(s.shape[0]*train_test_ratio)
        a_split_idx = int(a.shape[0]*train_test_ratio)
        self.train_s = s[:s_split_idx]
        self.test_s = s[s_split_idx:]
        self.train_a = a[:a_split_idx]
        self.test_a = a[a_split_idx:]


    def fit_model(self):
        if self.continuous:
            model = tree.DecisionTreeRegressor(max_depth=self.depth)
        else:
            model = tree.DecisionTreeClassifier(max_depth=self.depth)

        self.tree = model.fit(self.train_s, self.train_a)

        print('Tree fitted! Depth: {}  Accuracy:  {}'.format(self.tree.get_depth(), self.evaluate_accuracy(self.tree)))
        dump(self.tree, self.model_prefix+'decision_tree{}.joblib'.format(self.depth))

    def evaluate_accuracy(self, model):
        return model.score(self.test_s, self.test_a)

        
    def choose_action(self, s, DIST = False):
        s = np.array(s).reshape(1, -1)
        a = self.tree.predict(s)     
        if self.continuous:
            return a.reshape(-1), None 
        else:
            a = int(a)
            if DIST:
                prob = self.tree.predict_proba(s).reshape(-1)
                if 0 in prob:
                    prob = prob + 1e-20 # remove value 0 in categorical distribution, which can cause infinite KL-divergence
                dist = Categorical(torch.from_numpy(prob))
                return a, dist
            else:
                return a, None

    def get_feature_importances(self):
        return self.tree.feature_importances_


class Decision_tree_agent_3layer():  # discrete; not work well
    def __init__(self, env, Continuous):
        super(Decision_tree_agent_3layer, self).__init__()
        self.continuous = Continuous
    
    def choose_action(self, s, DIST = False):
        if s[3] <=-0.1:
            if s[1]<=0.998:
                if s[3]<=-0.266:
                    leaf_node=[22892, 28381, 527077, 28930]
                    a, dist = select_action_from_node(leaf_node)
                else:
                    leaf_node=[94200, 25061, 295280, 25158]
                    a, dist = select_action_from_node(leaf_node)
            else:
                if s[1]<=1.341:
                    leaf_node =  [186681, 3830, 5006, 3963]
                    a, dist=select_action_from_node(leaf_node)
                else:
                    leaf_node=[86403, 12734, 0, 12647]
                    a, dist=select_action_from_node(leaf_node)
        else:
            if s[6]<=0.5:
                if s[7]<=0.5:
                    leaf_node = [219460, 67289, 116486, 70692]
                    a, dist=select_action_from_node(leaf_node)
                else:
                    leaf_node=[1,0,0,0]
                    a, dist=select_action_from_node(leaf_node)
            else:
                leaf_node=[1,0,0,0]
                a, dist=select_action_from_node(leaf_node)
        if DIST:
            return a, dist
        else:
            return a, None


def demo_heuristic_lander(env, heuristic_agent, seed=None, render=False, collect_data=False):
    """
    Collect demonstrations.
    """
    env.seed(seed)
    total_reward_list=[]
    a_list=[]
    s_list=[]
    for i in range(10000):
        print('Episode: ', i)
        total_reward = 0
        steps = 0
        s = env.reset()
        while steps < 1000:
            a, _ = heuristic_agent.choose_action(s)
            s_list.append(s)
            a_list.append([a])
            s, r, done, info = env.step(a)
            total_reward += r

            if render and not collect_data :
                still_open = env.render()
                if still_open == False: break

            # if steps % 20 == 0 or done:
            #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if done: break

        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(i, total_reward, steps))

        total_reward_list.append(total_reward)  
    print('Average reward: {}'.format(np.mean(total_reward_list)))
    np.save('state', s_list)
    np.save('action', a_list)
    return total_reward


if __name__ == '__main__':
    # demo_heuristic_lander(LunarLander(), render=True)
    Continuous = False
    if Continuous:
        env = gym.make('LunarLanderContinuous-v2').unwrapped
    else:
        env = gym.make('LunarLander-v2').unwrapped
    heuristic_agent = Heuristic_agent(env, Continuous)
    # heuristic_agent = Simple_heuristic_agent(env, Continuous)
    # heuristic_agent = Decision_tree_agent(env, Continuous, depth=20)
    demo_heuristic_lander(env, heuristic_agent, render=True, collect_data = False)
    # data = np.load('sa.npy', allow_pickle=True)
    # print(np.array(data[0]).shape, np.array(data[1]).shape)
    
    
