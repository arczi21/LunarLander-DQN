from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import torch
from agent import Agent

"""
botforge code

link:
https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
"""


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('LunarLander-v2')
net = torch.load('model/trained_net.pkl')
agent = Agent(env, net, monitor=False)

frames = []

state = env.reset()
done = False
while not done:
    action = agent.greedy_action(state)
    frames.append(env.render(mode="rgb_array"))
    state, reward, done, _ = env.step(action)


env.close()
save_frames_as_gif(frames)


