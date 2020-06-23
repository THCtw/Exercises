import gym
from algs.vpg.vpg import vpg as vpg
from algs.vpg import core

env = gym.make('LunarLander-v2')
vpg(lambda: env)
