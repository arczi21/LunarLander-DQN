import numpy as np
import time
import cv2
import gym
import torch
from net import Net
from agent import Agent
import glob
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('LunarLander-v2')
net = torch.load('model/trained_net.pkl')
agent = Agent(env, net, monitor=False)


def render_episodes(n_episodes, render_every=3):
    frames = []
    idx = 0
    for e in range(n_episodes):
        state = env.reset()
        done = False
        r = 0
        while not done:
            action = agent.greedy_action(state)
            if idx % render_every == 0:
                frames.append(env.render(mode="rgb_array"))
            state, reward, done, _ = env.step(action)
            r += reward
            idx += 1
    return frames


f = np.array(render_episodes(5, 1), dtype=np.uint8)
f = [Image.fromarray(img) for img in f]
f[0].save("gym_animation.gif", save_all=True, append_images=f[1:])
env.close()
