import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import math, os
import cv2
import torchvision.transforms as transforms
import pybullet
import torchvision.models as models
import importlib

# from config import device
device = 'cuda' if torch.cuda.is_available () else 'cpu'

MAX_EP = 1000 * 1000
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.005

def v_wrap (np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype (dtype)
    return torch.from_numpy (np_array).to (device)


def push_and_pull (optim, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.
    else:
        v_s_ = lnet.forward (v_wrap (s_[None, :]))[-1].data.cpu ().numpy ()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:
        v_s_ = r
        buffer_v_target.append (v_s_)
    buffer_v_target.reverse ()

    loss = lnet.loss_func (
        v_wrap (np.array (bs)),
        v_wrap (np.array (ba)),
        v_wrap (np.array (buffer_v_target)[:, None]))

    # calculate local gradient and push local parameters to global
    optim.zero_grad ()
    loss.backward ()

    nn.utils.clip_grad_norm (lnet.parameters (), 1.0)
    for lp, gp in zip (lnet.parameters (), gnet.parameters ()):
        gp._grad = lp.grad
    optim.step ()

    # pull global parameters
    lnet.load_state_dict (gnet.state_dict ())


class SharedAdam (torch.optim.Adam):
    def __init__ (self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                  weight_decay=0):
        super (SharedAdam, self).__init__ (params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like (p.data)
                state['exp_avg_sq'] = torch.zeros_like (p.data)

                # share in memory
                state['exp_avg'].share_memory_ ()
                state['exp_avg_sq'].share_memory_ ()


def set_init (layers):
    for layer in layers:
        nn.init.normal_ (layer.weight, mean=0., std=0.01)
        nn.init.constant_ (layer.bias, 0.)


class ACNet (nn.Module):
    def __init__ (self):
        super (ACNet, self).__init__ ()
        self.distribution = torch.distributions.Normal
        self.state_dim = 24

        self.cnn = nn.Sequential (
            models.resnet18 (pretrained=True).to (device),
            nn.Linear (in_features=1000, out_features=512),
            nn.ReLU (),
            nn.Linear (in_features=512, out_features=256),
            nn.ReLU (),
        )

        # 2, 3
        self.fc_a = nn.Sequential (
            nn.Linear (256, 128),
            nn.ReLU (),
            nn.Linear (128, 64),
            nn.ReLU (),
            nn.Linear (64, 24),
            nn.ReLU ()
        )

        self.fc_s = nn.Sequential (
            nn.Linear (256, 128),
            nn.ReLU (),
            nn.Linear (128, 64),
            nn.ReLU (),
            nn.Linear (64, 24),
            nn.ReLU ()
        )

        self.fc_v = nn.Sequential (
            nn.Linear (256, 128),
            nn.ReLU (),
            nn.Linear (128, 64),
            nn.ReLU (),
            nn.Linear (64, 24),
            nn.ReLU ()
        )

        self.mu_layer = nn.Linear (24, 3)
        self.sigma_layer = nn.Linear (24, 3)
        self.v_layer = nn.Linear (24, 1)

        set_init ([self.mu_layer, self.sigma_layer, self.v_layer])

    def forward (self, im):
        im = torch.tensor (im).float ().squeeze ().to (device)

        if len(im.shape)==3:
            im = im.unsqueeze (0)

        im = self.cnn (im.transpose (1, 3))
        x_a = self.fc_a (im)
        mu = self.mu_layer (x_a)
        mu = F.tanh (mu)
        x_s = self.fc_s (im)
        sigma = self.sigma_layer (x_s)
        sigma = F.sigmoid (sigma) + 0.005
        x_v = self.fc_v (im)
        values = self.v_layer (x_v)
        return mu, sigma, values

    def choose_action (self, s):
        self.training = False
        mu, sigma, _ = self.forward (s)
        m = self.distribution (mu.view (-1, ).data, sigma.view (-1, ).data)
        return m.sample ().cpu ().numpy (), mu.cpu ().detach ().numpy (), sigma.cpu ().detach ().numpy ()

    def loss_func (self, s, a, v_t):
        self.train ()
        mu, sigma, values = self.forward (s)
        td = v_t - values
        c_loss = td.pow (2)

        m = self.distribution (mu, sigma)
        log_prob = m.log_prob (a)
        entropy = 0.5 + 0.5 * math.log (2 * math.pi) + torch.log (m.scale)
        exp_v = log_prob * td.detach () + ENTROPY_BETA * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean ()
        return total_loss


class Worker (mp.Process):
    def __init__ (self, gnet, optim, global_ep, global_ep_r, res_queue, wid, opt=None, RobotEnv=None,
                  SAVE_TOP_DIR=None):
        super (Worker, self).__init__ ()
        print ("wid %d" % wid)
        self.wid = wid
        self.step = 0

        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.optim = gnet, optim
        self.random_seed = 42 + self.wid + int (np.log (self.wid * 100 + 1))
        print ("random_seed", self.random_seed, "self.wid", self.wid)
        np.random.seed (self.random_seed)

        self.lnet = ACNet ().to (device)
        self.init_step = 0
        self.SAVE_TOP_DIR = SAVE_TOP_DIR
        self.RobotEnv = RobotEnv
        self.opt = opt

    def run (self):
        mean = np.array ([0.485, 0.456, 0.406])
        std = np.array ([0.229, 0.224, 0.225])
        mean = np.reshape (mean, (1, 1, 3))
        std = np.reshape (std, (1, 1, 3))

        show_id = -1
        if self.opt.gui:
            show_id = 0

        if self.wid == show_id:
            self.p_id = pybullet.connect (pybullet.GUI)
        else:
            self.p_id = pybullet.connect (pybullet.DIRECT)

        self.env = self.RobotEnv (worker_id=self.wid, opt=self.opt, p_id=pybullet)

        total_step = 1 + self.init_step
        suc_check = 0
        reward_check = 0
        episode_check = 0
        sigma_check1 = 0
        sigma_check2 = 0
        total_episode = 0

        buffer_s, buffer_a, buffer_r = [], [], []
        buffer_mem_s, buffer_mem_a, buffer_mem_r = [], [], []

        while self.g_ep.value < MAX_EP:
            observation = self.env.reset ()
            observation = observation / 255.0
            observation = (observation - mean) / std

            if self.wid == 0:
                print ("inital", np.sum (np.isnan (observation)), np.max (np.abs (observation)))

            while True:
                action, mu_r, sigma_r = self.lnet.choose_action (v_wrap (observation[None, :]))
                print ("action", action, "mu_r", mu_r, "sigma_r", sigma_r)
                self.env.info += 'action:{}\n'.format (str (action))
                self.env.info += 'mu:{}\n'.format (str (mu_r))
                self.env.info += 'sigma:{}\n'.format (str (sigma_r))

                action = action.clip (-0.2, 0.2)
                observation_next, reward, done, suc = self.env.step (action)
                observation_next = observation_next / 255.0
                observation_next = (observation_next - mean) / std

                buffer_s.append (observation)
                buffer_r.append (reward)
                buffer_a.append (action)

                if reward > 0.5:
                    buffer_mem_s.append (observation)
                    buffer_mem_a.append (action)
                    buffer_mem_r.append (reward)

                if total_step % (UPDATE_GLOBAL_ITER + self.wid) == 0:  # or self.g_ep.value % (10 * UPDATE_GLOBAL_ITER):
                    # sync
                    buffer_s = buffer_s + buffer_mem_s
                    buffer_a = buffer_a + buffer_mem_a
                    buffer_r = buffer_r + buffer_mem_r
                    push_and_pull (self.optim, self.lnet, self.gnet, done, observation_next, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if len (buffer_mem_s) > 20:
                        buffer_mem_s, buffer_mem_a, buffer_mem_r = [], [], []

                observation = observation_next
                total_step += 1
                reward_check += reward

                if self.env.epoch_num % 200 == 0:
                    save_path = os.path.join (self.env.log_root, str (total_step) + 'model.pth.tar')
                    torch.save (self.gnet.state_dict (), save_path)

                if done:
                    break


class A3C_solver:
    def __init__ (self, opt, RobotEnv):
        self.opt = opt
        self.RobotEnv = RobotEnv

    def run (self):
        mp.set_start_method ('spawn')
        gnet = ACNet ()  # .to(device)  # global network

        gnet.to (device)
        gnet.share_memory ()

        optim = SharedAdam (gnet.parameters (), lr=0.0002)
        global_ep, global_ep_r, res_queue = mp.Value ('i', 0), mp.Value ('d', 0.), mp.Queue ()

        workers = [Worker (gnet, optim, global_ep, global_ep_r, res_queue, i,
                           self.opt, self.RobotEnv) for i in range (self.opt.process_N)]
        [w.start () for w in workers]
        res = []

        for worker in workers:
            worker.init_step = 0

        while True:
            r = res_queue.get ()
            if r is not None:
                res.append (r)
            else:
                break
        [w.join () for w in workers]


if __name__ == "__main__":
    ExName = "55_v10"
    SAVE_TOP_DIR = os.path.join ('./saved_models', ExName, "entropy0.005_lr1e5_clip_s1.0")
    if not os.path.exists (SAVE_TOP_DIR):
        os.makedirs (SAVE_TOP_DIR)

    mp.set_start_method ('spawn')
    gnet = ACNet ()  # .to(device)  # global network

    ## loading
    Load_model_id = '10'
    Load_path = os.path.join (SAVE_TOP_DIR, Load_model_id + 'model.pth.tar')
    # checkpoint = torch.load(Load_path)
    # gnet.load_state_dict(checkpoint)

    gnet.to (device)
    gnet.share_memory ()

    optim = SharedAdam (gnet.parameters (), lr=0.0002)
    global_ep, global_ep_r, res_queue = mp.Value ('i', 0), mp.Value ('d', 0.), mp.Queue ()

    workers = [Worker (gnet, optim, global_ep, global_ep_r, res_queue, i, SAVE_TOP_DIR) for i in range (10)]
    [w.start () for w in workers]
    res = []

    for worker in workers:
        worker.init_step = 0

    while True:
        r = res_queue.get ()
        if r is not None:
            res.append (r)
        else:
            break
    [w.join () for w in workers]
