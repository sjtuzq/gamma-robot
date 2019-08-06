#!/usr/bin/env python3

"""
action 15: hit sth with sth

TODO : recover these functions: get_reward
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

class Engine15(Engine):
    def __init__(self,opt):
        super(Engine15,self).__init__(opt)
        self.opt = opt

    def init_grasp(self):
        pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

        for j in range (7):
            p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        for init_t in range(100):
            box = p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.05
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        start_id = 0
        init_traj = point2traj(points)
        start_id = self.move(init_traj,orn_traj,start_id)

        p.stepSimulation()

        # grasping
        grasp_stage_num = 10
        for grasp_t in range(grasp_stage_num):
            # self.gripperPos = get_gripper_pos (1-grasp_t/grasp_stage_num*0.5)
            self.gripperPos = get_gripper_pos (1-grasp_t/grasp_stage_num*0.7)
            p.setJointMotorControlArray (bodyIndex=self.robotId, jointIndices=self.activeGripperJointIndexList,
                                         controlMode=p.POSITION_CONTROL, targetPositions=self.gripperPos,
                                         forces=[self.gripperForce] * len (self.activeGripperJointIndexList))
            p.stepSimulation ()
            start_id += 1

        self.obj2_file = os.path.join (self.env_root, "urdf/objmodels/urdfs/cup.urdf")
        self.obj2_position = [0.45, 0.08, 0.34]
        # self.obj_position = [0.55, 0, 0.34]
        # self.obj_position = [0.40, -0.15, 0.34]
        self.obj2_orientation = p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        # self.obj_orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
        self.obj2_scaling = 0.11
        self.obj2_id = p.loadURDF (fileName=self.obj2_file, basePosition=self.obj2_position,
                                  baseOrientation=self.obj2_orientation,
                                  globalScaling=self.obj2_scaling, physicsClientId=self.physical_id)

        texture_path = os.path.join(self.env_root,'texture/sun_textures')
        texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
        textid = p.loadTexture (texture_file)
        # p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1, 1, 1, 0.9])
        p.changeVisualShape (self.obj2_id, -1, textureUniqueId=textid)
        self.start_pos = p.getLinkState (self.robotId, 7)[0]


    def get_reward (self):
        distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5

        obj2 = p.getAABB (self.obj2_id, -1)
        obj2_center = [(x + y) * 0.5 for x, y in zip (obj2[0], obj2[1])]
        obj = p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        aabb_dist = sum ([(x - y) ** 2 for x, y in zip (obj2_center, obj_center)]) ** 0.5

        reward = (0.5-aabb_dist)*100

        # calculate whether it is done
        self.info += 'now distance:{}\n'.format (distance)
        self.info += 'AABB distance:{}\n'.format (aabb_dist)

        if self.seq_num >= self.max_seq_num:
            done = True
        else:
            done = False

        for axis_dim in range (3):
            if self.start_pos[axis_dim] < self.axis_limit[axis_dim][0] or \
                    self.start_pos[axis_dim] > self.axis_limit[axis_dim][1]:
                done = True
                reward = -10

        if aabb_dist<0.1:
            done = True
            reward = 100

        # reward = -1
        self.info += 'reward: {}\n\n'.format (reward)
        self.log_info.write (self.info)
        print (self.info)
        return self.observation, reward, done, self.info
