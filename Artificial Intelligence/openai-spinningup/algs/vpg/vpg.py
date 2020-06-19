import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
import matplotlib.pyplot as plt

import gym

import algs.vpg.core as core

def vpg(env, record=1, policyAgent=core.PGAgent, gamma=0.5, seed=0, episodes_per_batch=5, num_batch=400):
    
    # Set up OpenAI Gym environment
    env = env()

    # Set up policy agent
    pgnet = core.PGNet()
    vfnet = core.VFNet
    agent = policyAgent(pgnet, vfnet)

    agent.pgnet.train()
    agent.vfnet.train()

    avg_total_rewards, avg_final_rewards = [], []
    
    prg_bar = tqdm(range(num_batch))
    for batch in prg_bar:

        states, log_probs, rewards, ret = [], [], [], []
        total_rewards, final_rewards = [], []

        # Collecting training data
        for episode in range(episodes_per_batch):

            state = env.reset()
            total_reward, total_step = 0, 0

            episode_rewards = []
            episode_states = []

            while True:

                action, log_prob = agent.sample(state)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                episode_states.append(state)
                episode_rewards.append(reward)

                state = next_state
                total_reward += reward
                total_step += 1

                if done:

                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    states.append(episode_states)
                    rewards.append(episode_rewards)
                    ret.append(core.discount_cumsum(rewards, gamma)
                    break

    	# Log training process
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")    

        # Update Policy Gradient Network
        agent.update_pgnet(log_probs, states, episodes_per_batch)
        ret = np.concatenate(ret, axis=0)
        agent.update_vfnet(states, torch.from_numpy(ret)) 

        plt.plot(avg_total_rewards)
        plt.title("Total Rewards")
        plt.savefig('avg_total_rewards.png')

        plt.plot(avg_final_rewards)
        plt.title("Final Rewards")
        plt.savefig('avg_final_rewards.png')

    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--episodes', type=int, default=4000)
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    vpg(lambda : gym.make(args.env), policyAgent=core.VPGAgent, gamma=args.gamma, seed=args.seed, episodes_per_batch=args.episodes, num_batch=batches) 
