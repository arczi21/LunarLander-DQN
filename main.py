import numpy as np
import time
import cv2
import gym
from net import Net
from agent import Agent

env = gym.make('LunarLander-v2')
net = Net()
agent = Agent(env, net)

for i in range(1000000):
    agent.step()
