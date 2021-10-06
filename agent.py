import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import time
import numpy as np
from collections import namedtuple, deque
from tensorboardX import SummaryWriter


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplay:
    def __init__(self, maxlen):
        self.replay = deque(maxlen=maxlen)

    def append(self, state, action, reward, next_state, done):
        exp = Experience(state, action, reward, next_state, done)
        self.replay.append(exp)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.replay, batch_size))

        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(self, env, net, monitor=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.net = net.to(self.device)
        self.target_net = copy.deepcopy(net).to(self.device)
        self.monitor = monitor

        self.EXPERIENCE_REPLAY_CAPACITY = 1000000
        self.EPSILON_START = 1.0
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 0.999
        self.BATCH_SIZE = 64
        self.MIN_EXPERIENCE_SIZE = self.BATCH_SIZE
        self.LEARNING_RATE = 0.001
        self.GAMMA = 0.99
        self.UPDATE_EVERY = 1000
        self.SAVE_EVERY = 1000

        self.experience_replay = ExperienceReplay(maxlen=self.EXPERIENCE_REPLAY_CAPACITY)
        self.state = self.env.reset()
        self.act_step = 1
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.cumulative_reward = 0.0
        self.episode = 0
        if self.monitor:
            self.writer = SummaryWriter(comment='v1')
        self.start = time.time()
        self.episode_len = 0
        self.losses = []
        self.epsilon = 1

        print(self.device)

    def greedy_action(self, state):
        state_t = torch.FloatTensor([state]).to(self.device)
        with torch.no_grad():
            vals = self.net(state_t)
        return torch.argmax(vals, dim=1).item()

    def calculate_epsilon(self):
        if self.epsilon < self.EPSILON_END:
            return self.EPSILON_END
        else:
            self.epsilon = self.epsilon * self.EPSILON_DECAY
            return self.epsilon

    def epsilon_greedy(self, state):
        epsilon = self.calculate_epsilon()
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.greedy_action(state)
        return action

    def step(self):
        action = self.epsilon_greedy(self.state)
        next_state, reward, done, _ = self.env.step(action)
        self.cumulative_reward += reward
        self.experience_replay.append(self.state, action, reward, next_state, done)

        self.state = next_state
        if done:
            print('Episode %d | Reward: %.3f | Step: %d | Fps: %.3f | Epsilon: %.3f' % (self.episode, self.cumulative_reward,
                                                                                  self.act_step, self.episode_len / (time.time() - self.start),
                                                                                  self.epsilon))
            self.start = time.time()
            if self.monitor:
                self.writer.add_scalar('reward', self.cumulative_reward, self.act_step)
                if self.act_step >= self.MIN_EXPERIENCE_SIZE:
                    self.writer.add_scalar('mean loss', np.mean(self.losses), self.act_step)
            self.losses = []
            self.episode += 1
            self.cumulative_reward = 0.0
            self.episode_len = 0
            self.state = self.env.reset()

        if self.act_step >= self.MIN_EXPERIENCE_SIZE:
            states, actions, rewards, next_states, dones = self.experience_replay.sample(self.BATCH_SIZE)
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            dones_t = torch.BoolTensor(dones).to(self.device)

            vals = self.net(states_t)
            vals = torch.gather(vals, 1, actions_t.view(-1, 1))

            with torch.no_grad():
                targets = self.target_net(next_states_t)
            targets = torch.max(targets, dim=1)[0]
            targets[dones_t] = 0.0
            targets = rewards_t + self.GAMMA * targets

            loss = self.criterion(vals, targets.view(-1, 1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

        if self.act_step % self.UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if self.act_step % self.SAVE_EVERY == 0:
            torch.save(self.net, 'model/net.pkl')

        self.act_step += 1
        self.episode_len += 1
