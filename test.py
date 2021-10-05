import numpy as np
import time
import cv2
import gym
import torch
from net import Net
from agent import Agent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('LunarLander-v2')
net = torch.load('model/trained_net.pkl')
agent = Agent(env, net, monitor=False)


def play_episodes(n_episodes):
    for e in range(n_episodes):
        state = env.reset()
        done = False
        r = 0
        while not done:
            action = agent.greedy_action(state)
            env.render()
            state, reward, done, _ = env.step(action)
            r += reward

        print('Episode %d, reward: %.3f' % (e + 1, r))


play_episodes(5)
env.close()
