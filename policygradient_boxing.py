import torch as th
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import gym
import io
from collections import namedtuple
import random
import math


class cnn(nn.Module):
    def __init__(s,udim=1):
        super().__init__()
        """
        con2d layers
        160 * 210 pixels to probability of 18 choices
        """
        s.conv1 = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6,6,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6,1,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.MaxPool2d(kernel_size=2),
        )

        s.linear_layers = nn.Sequential(
            nn.Linear(2080, udim)
        )

        s.std = 1

    def forward(s, x, eps, u=None):
        """
        This is a PyTorch function that runs the network
        on a state x to output a control u. 
        return the choice and the log probability
        """
        x = x.view(1,3,160,210)       
        x = s.conv1(x)
        x = x.view(1,-1)
        x = s.linear_layers(x)
        q_val = nn.Softmax(dim=1)(x)
        length = q_val.size()[1]
       
        pt = th.argmax(q_val)

        prob = th.ones(length)
        prob = prob / length * eps
        prob[pt] = prob[pt] + (1 - eps)

        m = th.distributions.Multinomial(length, prob)

        if u is None:
            u = m.rsample()
        logp = m.log_prob(u)
        
        return u, logp

def rollout(policy, env, eps):
    """
    We will use the control u_theta(x_t) to take the control action at each
    timestep. You can use simple Euler integration to simulate the ODE forward
    for T = 200 timesteps with discretization dt=0.05.
    At each time-step, you should record the state x, control u, and the reward r.
    """
    thisstep = 1
    us = []
    rs = []
    gamma = 0.99
    for step in range(1000):
        env.render()
        
        obs, reward, done, info = env.step(thisstep)
        obs = obs.astype('float32')
        state = th.from_numpy(obs.T)

        # state = state.view(1,3,160,210)
        u, logp = policy(state, eps)
        thisstep = th.argmax(u)
        # plt.imshow(jpgname,status)
        # print(reward)
        if done: 
            print('dead in %d steps' % step)
            break

        rs.append(reward)
        us.append(logp)


    # R is the discounted cumulative reward
    R = sum([rr * gamma ** k for k, rr in enumerate(rs)])
    return {'u': th.tensor(us).float(),
            'r': th.tensor(rs).float(),
            'R': R}



env = gym.make('Boxing-v0')
udim = env.action_space.n
status = env.reset()

policy = cnn(udim)
optim = th.optim.Adam(policy.parameters(), lr=1e-3)

num_ite = 1000
num_traj = 2
rew = np.zeros(num_ite)

eps = 0.9

for k in range(num_ite):
    '''
    1. Get a trajectory
    '''
    t = []
    for i in range(num_traj):
        t.append(rollout(policy, env, eps))

    """"
    2. We now want to calculate grad log u_theta(u | x), so
    we will feed all the states from the trajectory again into the network
    and this time we are interested in the log-probabilities. The following
    code shows how to update the weights of the model using one trajectory
    """
    mm = 0
    for i in range(num_traj):
        mm = t[i]['R']/num_traj + mm

    f = 0
    for i in range(num_traj):
        logp = t[i]['u']
        f = -((t[i]['R'] - mm) * logp).mean()/num_traj + f

    # .zero_grad() is a PyTorch peculiarity that clears the backpropagation
    # gradient buffer before calling the next .backward()
    policy.zero_grad()
    # .backward() computes the gradient of the policy gradient objective with respect
    # to the parameters of the policy and stores it in the gradient buffer
    f.backward()
    # .step() updates the weights of the policy using the computed gradient
    optim.step()

    rew[k] = mm

    if (k%10 == 0):
        eps = max(eps*0.95, 0.1)

plt.plot(range(num_ite),rew)
plt.savefig('./test3.jpg')
plt.show()


