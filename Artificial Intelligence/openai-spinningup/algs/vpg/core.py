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

