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

def vpg(env, record=1, policyAgent=core.PGAgent, gamma=0.9, seed=0, EPISODE_PER_BATCH=50, num_batch=50, train_v_iters=80):
    
    # Set up OpenAI Gym environment
    env = env()

    # Set up policy agent
    pgnet = core.PGNet()
    vfnet = core.VFNet()
    agent = policyAgent(pgnet, vfnet)

    agent.pgnet.train()
    agent.vfnet.train()

    avg_total_rewards, avg_final_rewards = [], []
    
    prg_bar = tqdm.tqdm(total=num_batch)
    for batch in range(num_batch):

        states, log_probs, rewards, ret = [], [], [], []
        total_rewards, final_rewards = [], []

        # Collecting training data
        for episode in range(EPISODE_PER_BATCH):

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
                    ret.append(core.discount_cumsum(episode_rewards, gamma))
                    break

    	# Log training process
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")    
        prg_bar.update(1)

        # Update Policy Gradient Network
        states = np.concatenate(states, axis=0)
        ret = np.concatenate(ret, axis=0)
        agent.update_pgnet(log_probs, states, EPISODE_PER_BATCH)
        agent.learn_vfnet(states, torch.from_numpy(ret), train_v_iters) 

    plt.plot(avg_total_rewards)
    plt.title("Total Rewards")
    plt.savefig('results/avg_total_rewards.png')
    plt.close()

    plt.plot(avg_final_rewards)
    plt.title("Final Rewards")
    plt.savefig('avg_final_rewards.png')
    plt.close()


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
