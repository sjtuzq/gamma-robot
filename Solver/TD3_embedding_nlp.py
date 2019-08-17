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

device = 'cuda' if torch.cuda.is_available () else 'cpu'

'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''


class Replay_buffer ():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__ (self, max_size,opt=None):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.opt = opt
        self.embedding_data = np.load (
            os.path.join (self.opt.project_root,'scripts', 'utils',
                          'labels', 'label_embedding_uncased.npy'))
        if self.opt.test_id==3000:
            self.embedding_data = np.load (
                os.path.join (self.opt.project_root, 'scripts', 'utils',
                              'labels', 'label_embedding_uncased_adjusted.npy'))
        if self.opt.test_id==121:
            self.embedding_data = np.load (
                os.path.join (self.opt.project_root, 'scripts', 'utils',
                              'labels', 'label_embedding_uncased_adjusted.npy'))

    def push (self, data):
        if len (self.storage) == self.max_size:
            self.storage[int (self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append (data)

    def sample (self, batch_size):
        ind = np.random.randint (0, len (self.storage), size=batch_size)
        x_embedding, x, y_embedding, y, u, r, d = [], [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x_embedding.append (np.array (X[0], copy=False))
            x.append (np.array (X[1], copy=False))

            y_embedding.append (np.array (Y[0], copy=False))
            y.append (np.array (Y[1], copy=False))

            u.append (np.array (U, copy=False))
            if self.opt.nlp_embedding:
                reward_id = np.where ((Y[0] == self.embedding_data).sum (1) == 1024)[0][0]
                reward_id = np.where(np.array(self.opt.embedding_list) == reward_id)[0][0]
                r.append (np.array (R[reward_id], copy=False))
            else:
                r.append (np.array (R[np.where(Y[0]==1)[0][0]], copy=False))
            d.append (np.array (D, copy=False))

        return np.array (x_embedding), np.array (x), np.array (y_embedding), np.array (y), np.array (u), np.array (
            r).reshape (-1, 1), np.array (d).reshape (-1, 1)

    def align_sample (self, batch_size):
        ind = np.random.randint (0, len (self.storage), size=int(batch_size/len(self.opt.embedding_list)))
        x_embedding, x, y_embedding, y, u, r, d = [], [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x_embedding.append (np.array (X[0], copy=False))
            x.append (np.array (X[1], copy=False))

            y_embedding.append (np.array (Y[0], copy=False))
            y.append (np.array (Y[1], copy=False))

            u.append (np.array (U, copy=False))
            if self.opt.nlp_embedding:
                reward_id_for_action = np.where ((Y[0] == self.embedding_data).sum (1) == 1024)[0][0]
                reward_id = np.where (np.array (self.opt.embedding_list) == reward_id_for_action)[0][0]
                r.append (np.array (R[reward_id], copy=False))
                d.append (np.array (D, copy=False))

                # conjugated sample
                for conj_id in self.opt.embedding_list:
                    if conj_id == reward_id_for_action:
                        continue
                    align_id_for_action = conj_id
                    align_embedding = self.embedding_data[align_id_for_action]
                    x_embedding.append (np.array (align_embedding, copy=False))
                    x.append (np.array (X[1], copy=False))

                    align_id_for_action = conj_id
                    align_embedding = self.embedding_data[align_id_for_action]
                    y_embedding.append (np.array (align_embedding, copy=False))
                    y.append (np.array (Y[1], copy=False))

                    u.append (np.array (U, copy=False))
                    reward_id = np.where (np.array (self.opt.embedding_list) == align_id_for_action)[0][0]
                    r.append (np.array (R[reward_id], copy=False))
                    d.append (np.array (D, copy=False))
            else:
                r.append (np.array (R[np.where (Y[0] == 1)[0][0]], copy=False))
                d.append (np.array (D, copy=False))

                # conjugated sample
                for conj_id in np.where (Y[0] == 0)[0]:
                    conj_X = np.zeros_like (X[0])
                    conj_X[conj_id] = 1
                    x_embedding.append (np.array (conj_X, copy=False))
                    x.append (np.array (X[1], copy=False))

                    conj_Y = np.zeros_like (Y[0])
                    conj_Y[conj_id] = 1
                    y_embedding.append (np.array (conj_Y, copy=False))
                    y.append (np.array (Y[1], copy=False))

                    u.append (np.array (U, copy=False))
                    r.append (np.array (R[conj_id], copy=False))
                    d.append (np.array (D, copy=False))

        return np.array (x_embedding), np.array (x), np.array (y_embedding), np.array (y), np.array (u), np.array (
            r).reshape (-1, 1), np.array (d).reshape (-1, 1)


class Actor (nn.Module):

    def __init__ (self, state_dim, action_dim, max_action, opt):
        super (Actor, self).__init__ ()

        self.opt = opt

        self.embedding_dim = self.opt.rl_embedding_dim
        self.feature_fc = nn.Sequential(
            nn.Linear (self.opt.embedding_dim, 512),
            nn.ReLU (),
            nn.Linear (512, 256),
            nn.ReLU (),
            nn.Linear (256, 256),
            nn.ReLU (),
            nn.Linear (256, 256),
        )
        self.nlp_fc =  nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.embedding_dim),
        )
        self.stn = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear (256, 256),
            nn.ReLU (),
            nn.Linear(256,self.embedding_dim*self.embedding_dim),
        )
        self.fc1 = nn.Linear (state_dim + self.embedding_dim, 400)

        if not self.opt.more_embedding:
            self.embedding_dim = 0

        self.fc2 = nn.Sequential(
            nn.Linear (400 + self.embedding_dim, 300),
            nn.ReLU(),
            nn.Linear(300,200)
        )
        # self.fc3 = nn.Linear (300 + self.embedding_dim, 1)
        self.fc3 = nn.Sequential(
            nn.Linear (200 + self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,action_dim)
        )

        self.max_action = max_action

        self.cnn = nn.Sequential (
            models.resnet18 (pretrained=True).to (device),
            nn.Linear (in_features=1000, out_features=512),
            nn.ReLU (),
            nn.Linear (in_features=512, out_features=state_dim),
            nn.ReLU (),
        )

    def forward (self, action_embedding, state):
        if self.opt.observation == 'before_cnn':
            state = self.cnn (state.transpose (1, 3))

        action_embedding_feature = self.feature_fc(action_embedding)

        action_embedding = self.nlp_fc(action_embedding_feature)
        action_stn = self.stn(action_embedding_feature).view(-1,self.embedding_dim,self.embedding_dim)

        action_embedding = torch.bmm(action_embedding.unsqueeze(1),action_stn).squeeze(1)

        state = torch.cat ([action_embedding, state], 1)
        a = F.relu (self.fc1 (state))

        if self.opt.more_embedding:
            a = torch.cat([action_embedding,a],1)
        a = F.relu (self.fc2 (a))

        if self.opt.more_embedding:
            a = torch.cat([action_embedding,a],1)

        a = torch.tanh (self.fc3 (a)) * self.max_action
        return a


class Critic (nn.Module):

    def __init__ (self, state_dim, action_dim, opt):
        super (Critic, self).__init__ ()

        self.opt = opt

        self.embedding_dim = self.opt.rl_embedding_dim
        self.feature_fc = nn.Sequential(
            nn.Linear (self.opt.embedding_dim, 512),
            nn.ReLU (),
            nn.Linear (512, 256),
            nn.ReLU (),
            nn.Linear (256, 256),
            nn.ReLU (),
            nn.Linear (256, 256),
        )
        self.nlp_fc =  nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.embedding_dim),
        )
        self.stn = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear (256, 256),
            nn.ReLU (),
            nn.Linear(256,self.embedding_dim*self.embedding_dim),
        )
        self.fc1 = nn.Linear (state_dim + action_dim + self.embedding_dim, 400)

        if not self.opt.more_embedding:
            self.embedding_dim = 0

        self.fc2 = nn.Sequential(
            nn.Linear (400 + self.embedding_dim, 300),
            nn.ReLU(),
            nn.Linear(300,200)
        )
        # self.fc3 = nn.Linear (300 + self.embedding_dim, 1)
        self.fc3 = nn.Sequential(
            nn.Linear (200 + self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

        self.cnn = nn.Sequential (
            models.resnet18 (pretrained=True).to (device),
            nn.Linear (in_features=1000, out_features=512),
            nn.ReLU (),
            nn.Linear (in_features=512, out_features=state_dim),
            nn.ReLU (),
        )

    def forward (self, action_embedding, state, action):
        if self.opt.observation == 'before_cnn':
            state = self.cnn (state.transpose (1, 3))

        # action_embedding = self.nlp_fc(action_embedding)

        action_embedding_feature = self.feature_fc(action_embedding)
        action_embedding = self.nlp_fc(action_embedding_feature)
        action_stn = self.stn(action_embedding_feature).view(-1,self.embedding_dim,self.embedding_dim)
        action_embedding = torch.bmm(action_embedding.unsqueeze(1),action_stn).squeeze(1)

        # state_action = torch.cat([state, action], 1)
        state_action = torch.cat ([action_embedding, state, action], 1)
        q = F.relu (self.fc1 (state_action))

        if self.opt.more_embedding:
            q = torch.cat([action_embedding,q],1)

        q = F.relu (self.fc2 (q))
        if self.opt.more_embedding:
            q = torch.cat([action_embedding,q],1)

        q = self.fc3 (q)
        return q


class TD3_embedding_nlp ():
    def __init__ (self, state_dim, action_dim, max_action, directory, opt):

        self.args = opt
        self.directory = directory

        self.actor = Actor (state_dim, action_dim, max_action, opt).to (device)
        self.actor_target = Actor (state_dim, action_dim, max_action, opt).to (device)
        self.critic_1 = Critic (state_dim, action_dim, opt).to (device)
        self.critic_1_target = Critic (state_dim, action_dim, opt).to (device)
        self.critic_2 = Critic (state_dim, action_dim, opt).to (device)
        self.critic_2_target = Critic (state_dim, action_dim, opt).to (device)

        self.actor_optimizer = optim.Adam (self.actor.parameters (), lr=self.args.learning_rate)
        self.critic_1_optimizer = optim.Adam (self.critic_1.parameters (), lr=self.args.learning_rate)
        self.critic_2_optimizer = optim.Adam (self.critic_2.parameters (), lr=self.args.learning_rate)

        self.actor_target.load_state_dict (self.actor.state_dict ())
        self.critic_1_target.load_state_dict (self.critic_1.state_dict ())
        self.critic_2_target.load_state_dict (self.critic_2.state_dict ())

        self.max_action = max_action
        self.memory = Replay_buffer (self.args.capacity,self.args)
        self.writer = SummaryWriter (self.directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action (self, state):
        action_embedding = state[0]
        state = state[1]
        if self.args.observation == 'before_cnn':
            state = torch.tensor (state).float ().squeeze ().unsqueeze (0).to (device)
        else:
            state = torch.tensor (state.reshape (1, -1)).float ().to (device)

        action_embedding = torch.tensor (action_embedding).float ().unsqueeze (0).to (device)
        return self.actor (action_embedding, state).cpu ().data.numpy ().flatten ()

    def update (self, num_iteration):

        if self.num_training % 500 == 0:
            print ("====================================")
            print ("model has been trained for {} times...".format (self.num_training))
            print ("====================================")
        for i in range (num_iteration):
            if self.args.align_sample:
                x_embedding, x, y_embedding, y, u, r, d = self.memory.align_sample (self.args.batch_size)
            else:
                x_embedding, x, y_embedding, y, u, r, d = self.memory.sample (self.args.batch_size)

            action_embedding = torch.FloatTensor (x_embedding).to (device)
            state = torch.FloatTensor (x).to (device)

            next_action_embedding = torch.FloatTensor (y_embedding).to (device)
            next_state = torch.FloatTensor (y).to (device)

            action = torch.FloatTensor (u).to (device)
            done = torch.FloatTensor (d).to (device)
            reward = torch.FloatTensor (r).to (device)

            # Select next action according to target policy:
            noise = torch.ones_like (action).data.normal_ (0, self.args.each_action_lim * self.args.policy_noise).to (
                device)
            noise = noise.clamp (-self.args.noise_clip, self.args.noise_clip)
            next_action = (self.actor_target (next_action_embedding, next_state) + noise)
            next_action = next_action.clamp (-self.max_action, self.max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target (next_action_embedding, next_state, next_action)
            target_Q2 = self.critic_2_target (next_action_embedding, next_state, next_action)
            target_Q = torch.min (target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.args.gamma * target_Q).detach ()

            # Optimize Critic 1:
            current_Q1 = self.critic_1 (action_embedding, state, action)
            loss_Q1 = F.mse_loss (current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad ()
            loss_Q1.backward ()
            self.critic_1_optimizer.step ()
            self.writer.add_scalar ('Loss/Q1_loss', loss_Q1, global_step=self.num_critic_update_iteration)

            # Optimize Critic 2:
            current_Q2 = self.critic_2 (action_embedding, state, action)
            loss_Q2 = F.mse_loss (current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad ()
            loss_Q2.backward ()
            self.critic_2_optimizer.step ()
            self.writer.add_scalar ('Loss/Q2_loss', loss_Q2, global_step=self.num_critic_update_iteration)

            # Delayed policy updates:
            if i % self.args.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1 (action_embedding, state, self.actor (action_embedding, state)).mean ()

                # Optimize the actor
                self.actor_optimizer.zero_grad ()
                actor_loss.backward ()
                self.actor_optimizer.step ()

                self.writer.add_scalar ('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
                for param, target_param in zip (self.actor.parameters (), self.actor_target.parameters ()):
                    target_param.data.copy_ (((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip (self.critic_1.parameters (), self.critic_1_target.parameters ()):
                    target_param.data.copy_ (((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                for param, target_param in zip (self.critic_2.parameters (), self.critic_2_target.parameters ()):
                    target_param.data.copy_ (((1 - self.args.tau) * target_param.data) + self.args.tau * param.data)

                self.num_actor_update_iteration += 1

        self.num_critic_update_iteration += 1
        self.num_training += 1

    def save (self,epoch_id):
        torch.save (self.actor.state_dict (), os.path.join (self.directory, 'actor_{}.pth'.format(epoch_id)))
        torch.save (self.actor_target.state_dict (), os.path.join (self.directory, 'actor_target_{}.pth'.format(epoch_id)))
        torch.save (self.critic_1.state_dict (), os.path.join (self.directory, 'critic_1_{}.pth'.format(epoch_id)))
        torch.save (self.critic_1_target.state_dict (), os.path.join (self.directory, 'critic_1_target_{}.pth'.format(epoch_id)))
        torch.save (self.critic_2.state_dict (), os.path.join (self.directory, 'critic_2_{}.pth'.format(epoch_id)))
        torch.save (self.critic_2_target.state_dict (), os.path.join (self.directory, 'critic_2_target_{}.pth'.format(epoch_id)))
        print ("====================================")
        print ("Model has been saved...")
        print ("====================================")

    def load (self,epoch_id):
        self.actor.load_state_dict (torch.load (os.path.join (self.directory, 'actor_{}.pth'.format(epoch_id))))
        self.actor_target.load_state_dict (torch.load (os.path.join (self.directory, 'actor_target_{}.pth'.format(epoch_id))))
        self.critic_1.load_state_dict (torch.load (os.path.join (self.directory, 'critic_1_{}.pth'.format(epoch_id))))
        self.critic_1_target.load_state_dict (torch.load (os.path.join (self.directory, 'critic_1_target_{}.pth'.format(epoch_id))))
        self.critic_2.load_state_dict (torch.load (os.path.join (self.directory, 'critic_2_{}.pth'.format(epoch_id))))
        self.critic_2_target.load_state_dict (torch.load (os.path.join (self.directory, 'critic_2_target_{}.pth'.format(epoch_id))))
        print ("====================================")
        print ("model has been loaded...")
        print ("====================================")
