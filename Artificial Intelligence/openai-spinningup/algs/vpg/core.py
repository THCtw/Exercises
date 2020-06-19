import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def discount_cumsum(x, discount):
	"""
    Computing discounted cumulative sums of vectors
	
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
        self.fc3 = nn.Linear(16, 4)
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

    def update_pgnet(self, log_probs, states, episodes_per_batch):
        advs = self.vfnet(torch.FloatTensor(states))

        log_probs = torch.stack(log_probs)
        advs = torch.stack(advs)
        loss = (-log_probs * advs).mean()

        self.pg_optimizer.zero_grad()
        loss.backward()
        self.pg_optimizer.step()

    def update_vfnet(self, states, ret):
        vals = self.vfnet(torch.FloatTensor(states)
        vals = torch.stack(vals)
        loss = ((vals - ret)**2).mean()
        
        self.vf_optimizer.zero_grad()
        loss.backward()
        self.vf_optimizer.step()

    def sample(self, state):
        action_prob = self.pgnet(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
