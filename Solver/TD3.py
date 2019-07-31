import argparse
from collections import namedtuple
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'



'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action,opt):
        super(Actor, self).__init__()

        self.opt = opt

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

        self.cnn = nn.Sequential(
            models.resnet18 (pretrained=True).to(device),
            nn.Linear (in_features=1000, out_features=512),
            nn.ReLU (),
            nn.Linear (in_features=512, out_features=state_dim),
            nn.ReLU (),
        )

    def forward(self, state):
        if self.opt.observation == 'before_cnn':
            state = self.cnn(state.transpose (1, 3))
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim,opt):
        super(Critic, self).__init__()

        self.opt = opt

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.cnn = nn.Sequential(
            models.resnet18(pretrained=True).to(device),
            nn.Linear (in_features=1000, out_features=512),
            nn.ReLU (),
            nn.Linear (in_features=512, out_features=state_dim),
            nn.ReLU (),
        )

    def forward(self, state, action):
        if self.opt.observation == 'before_cnn':
            state = self.cnn(state.transpose (1, 3))

        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self, state_dim, action_dim, max_action,directory,opt):
        
        self.args = opt
        self.directory = directory

        self.actor = Actor(state_dim, action_dim, max_action, opt).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, opt).to(device)
        self.critic_1 = Critic(state_dim, action_dim, opt).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, opt).to(device)
        self.critic_2 = Critic(state_dim, action_dim, opt).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, opt).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_action = max_action
        self.memory = Replay_buffer(self.args.capacity)
        self.writer = SummaryWriter(self.directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        if self.args.observation == 'before_cnn':
            state = torch.tensor(state).float().squeeze().unsqueeze(0).to(device)
        else:
            state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, num_iteration):

        if self.num_training % 500 == 0:
            print("====================================")
            print("model has been trained for {} times...".format(self.num_training))
            print("====================================")
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.args.each_action_lim*self.args.policy_noise).to(device)
            noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.args.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            self.writer.add_scalar('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            self.writer.add_scalar('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)
            # Delayed policy updates:
            if i % self.args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), os.path.join(self.directory,'actor.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(self.directory,'actor_target.pth'))
        torch.save(self.critic_1.state_dict(), os.path.join(self.directory,'critic_1.pth'))
        torch.save(self.critic_1_target.state_dict(), os.path.join(self.directory,'critic_1_target.pth'))
        torch.save(self.critic_2.state_dict(), os.path.join(self.directory,'critic_2.pth'))
        torch.save(self.critic_2_target.state_dict(), os.path.join(self.directory,'critic_2_target.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(os.path.join(self.directory, 'actor.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(self.directory, 'actor_target.pth')))
        self.critic_1.load_state_dict(torch.load(os.path.join(self.directory, 'critic_1.pth')))
        self.critic_1_target.load_state_dict(torch.load(os.path.join(self.directory, 'critic_1_target.pth')))
        self.critic_2.load_state_dict(torch.load(os.path.join(self.directory, 'critic_2.pth')))
        self.critic_2_target.load_state_dict(torch.load(os.path.join(self.directory, 'critic_2_target.pth')))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
