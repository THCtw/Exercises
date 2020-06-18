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

class VPGAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def learn(self, log_probs, rewards, EPISODE_PER_BATCH):
        loss = (-log_probs * rewards).sum() / EPISODE_PER_BATCH

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
