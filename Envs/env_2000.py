#!/usr/bin/env python3
"""
   env for action 2000: move sth up + move sth down
"""
import pybullet as p
import time
import math
from datetime import datetime
from time import sleep
import numpy as np
import random
import pybullet_data
import cv2
import os
import argparse
import torch

import sys
sys.path.append('./Eval')
sys.path.append('./')
from .utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code


import pkgutil
egl = pkgutil.get_loader ('eglRenderer')


from .env import Engine

class Engine2000(Engine):
    def __init__(self,opt):
        super(Engine2000,self).__init__(opt)

    def init_grasp(self):
        p.resetBasePositionAndOrientation (self.table_id, [0.42, 0, 0], [0, 0, math.pi * 0.32, 1])
        pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        self.robot.gripperControl(0)

        for init_t in range(100):
            box = p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.06
            center[1] -= 0.06
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        start_id = 0
        init_traj = point2traj(points)
        start_id = self.move(init_traj,orn_traj,start_id)

        # grasping
        grasp_stage_num = 10
        for grasp_t in range(grasp_stage_num):
            gripperPos = grasp_t / float(grasp_stage_num) * 180.0
            self.robot.gripperControl(gripperPos)
            start_id += 1

        pos = p.getLinkState (self.robotId, 7)[0]
        # up_traj = point2traj([pos, [pos[0]-0.1, pos[1]+0.08, pos[2]+0.1]])
        up_traj = point2traj([pos, [pos[0]-0.1, pos[1]+0.08, pos[2]+0.2]])
        start_id = self.move(up_traj, orn_traj,start_id)
        p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

        self.start_pos = p.getLinkState (self.robotId, 7)[0]

        p.resetBasePositionAndOrientation(self.table_id, [0.42,0,-0.14],[0,0,math.pi*0.32,1])

    def get_handcraft_reward (self):
        distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5
        # box = p.getAABB (self.box_id, -1)
        # box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        # obj = p.getAABB (self.obj_id, -1)
        # obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        # aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

        if self.opt.reward_diff:
            reward = (self.last_distance - distance) * 100
        else:
            reward = (0.5-distance)*100

        # calculate whether it is done
        self.info += 'now distance:{}\n'.format (distance)
        # self.info += 'AABB distance:{}\n'.format (aabb_dist)

        if self.seq_num >= self.max_seq_num:
            done = True
        else:
            done = False

        # check whether the object is still in the gripper
        left_closet_info = p.getContactPoints (self.robotId, self.obj_id, 13, -1)
        right_closet_info = p.getContactPoints (self.robotId, self.obj_id, 17, -1)
        if self.opt.obj_away_loss:
            if len (left_closet_info) == 0 and len (right_closet_info) == 0:
                done = True

        # reward = -1
        self.info += 'reward: {}\n\n'.format (reward)
        # self.log_info.write (self.info)
        # print (self.info)
        return self.observation, reward, done, self.info
