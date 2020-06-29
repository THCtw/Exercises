import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def discount_cumsum(x, discount):
	"""

	
	input: 
		vector x, [x0, x1, x2]

	output:
		[x0 + discount * x1 + discount^2 * x2,
		 x1 + discount * x2,
		 x2]
	"""
	return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PGNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)

class VFNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        hid = torch.tanh(self.fc3(hid))
        return self.fc4(hid)
    
class PGAgent():

    def __init__(self, pgnet, vfnet):
        self.pgnet = pgnet
        self.vfnet = vfnet
        self.pg_optimizer = optim.SGD(self.pgnet.parameters(), lr=0.001)
        self.vf_optimizer = optim.SGD(self.vfnet.parameters(), lr=0.001)

    def update_pgnet(self, log_probs, states, EPISODE_PER_BATCH):
        log_probs = torch.stack(log_probs)
        advs = self.vfnet(torch.FloatTensor(states))
        # advs = torch.FloatTensor(states)
        advs = (advs - advs.mean()) / advs.std()
        loss = (-log_probs * advs).sum()/EPISODE_PER_BATCH

        self.pg_optimizer.zero_grad()
        loss.backward()
        self.pg_optimizer.step()

    def learn_vfnet(self, states, rewards_to_go, train_v_iters):
        for i in range(train_v_iters):
            value_estimates = self.vfnet(torch.FloatTensor(states))
            #(value_estimates - value_estimates.mean()) / value_estimates.std()
            loss = ((value_estimates-rewards_to_go)**2).mean()
            if (i % 10) == 0: 
                print("vfnet.loss = {}".format(loss))

            self.vf_optimizer.zero_grad()
            loss.backward()
            self.vf_optimizer.step()

    def sample(self, state):
        action_prob = self.pgnet(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
