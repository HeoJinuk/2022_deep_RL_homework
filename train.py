import argparse
import os
import gym
from model import A3CAgent


# Training Parameters
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: BreakoutDeterministic-v4)')

if __name__ == "__main__":
    args = parser.parse_args()
    
    env = gym.make(args.env_name)
    action_size = env.action_space.n
    global_agent = A3CAgent(action_size=action_size, env_name=args.env_name,
                            discount_factor=args.gamma, 
                            t_max=args.num_steps, lr=args.lr, threads=args.num_processes,
                            num_episode=args.max_episode_length)
    global_agent.train()