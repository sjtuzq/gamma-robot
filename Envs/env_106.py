#!/usr/bin/env python3

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
from .model import CNN

import pkgutil
egl = pkgutil.get_loader ('eglRenderer')


from .env import Engine

class Engine106(Engine):
    def __init__(self,opt):
        super(Engine106,self).__init__(opt)
        self.opt = opt

    def init_grasp(self):
        pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

        for j in range (7):
            p.resetJointState(self.kukaId, j, self.data_q[0][j], self.data_dq[0][j])

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
        start_id = self.core(init_traj,orn_traj,start_id)

        p.stepSimulation()

        # grasping
        grasp_stage_num = 10
        for grasp_t in range(grasp_stage_num):
            # self.gripperPos = get_gripper_pos (1-grasp_t/grasp_stage_num*0.5)
            self.gripperPos = get_gripper_pos (1-grasp_t/grasp_stage_num*0.7)
            p.setJointMotorControlArray (bodyIndex=self.kukaId, jointIndices=self.activeGripperJointIndexList,
                                         controlMode=p.POSITION_CONTROL, targetPositions=self.gripperPos,
                                         forces=[self.gripperForce] * len (self.activeGripperJointIndexList))
            p.stepSimulation ()
            start_id += 1

        pos = p.getLinkState (self.kukaId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+0.3]])
        start_id = self.core(up_traj, orn_traj,start_id)

        if self.opt.rand_start == 'rand':
            # move in z-axis direction
            pos = p.getLinkState (self.kukaId, 7)[0]
            up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
            start_id = self.core(up_traj, orn_traj,start_id)

            # move in y-axis direction
            pos = p.getLinkState (self.kukaId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
            start_id = self.core (up_traj, orn_traj, start_id)

            # move in x-axis direction
            pos = p.getLinkState (self.kukaId, 7)[0]
            up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
            start_id = self.core (up_traj, orn_traj, start_id)

        elif self.opt.rand_start == 'two':
            prob = random.random()
            if prob<0.5:
                pos = p.getLinkState (self.kukaId, 7)[0]
                up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
                start_id = self.core (up_traj, orn_traj, start_id)

        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45, -0.1, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling, physicsClientId=self.physical_id)

        texture_path = os.path.join(self.env_root,'texture/sun_textures')
        texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
        textid = p.loadTexture (texture_file)
        p.changeVisualShape (self.plane_id, -1, rgbaColor=[1, 1, 1, 0.9])
        # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
        self.start_pos = p.getLinkState (self.kukaId, 7)[0]

        box = p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        obj = p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    def get_reward (self):
        distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5

        box = p.getAABB (self.box_id, -1)
        box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        obj = p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

        if self.opt.reward_diff:
            reward = (self.last_aabb_dist - aabb_dist) * 100
        else:
            reward = (0.5-aabb_dist)*100

        # reward = (self.last_aabb_dist-aabb_dist)*100
        self.last_aabb_dist = aabb_dist

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
                reward = self.opt.out_reward


        left_closet_info = p.getContactPoints (self.kukaId, self.obj_id, 13, -1)
        right_closet_info = p.getContactPoints (self.kukaId, self.obj_id, 17, -1)

        # left_closet_info = p.getClosestPoints(self.kukaId,self.obj_id,0.3,13,-1)
        # right_closet_info = p.getClosestPoints(self.kukaId,self.obj_id,0.3,17,-1)

        # print(left_closet_info,right_closet_info)
        # print(left_closet_info[0][8],right_closet_info[0][8],p.getLinkState(self.kukaId,13)[0],p.getLinkState(self.kukaId,17)[0])

        # if len (left_closet_info) > 0:
        #     print ('13:', left_closet_info[0][8])
        # if len (right_closet_info) > 0:
        #     print ('17:', right_closet_info[0][8])

        if self.opt.obj_away_loss:
            if len (left_closet_info)==0 and len (right_closet_info)==0:
                done = True
                reward = self.opt.out_reward

        if aabb_dist<self.opt.end_distance:
            done = True
            reward = 100

        # reward = -1
        self.info += 'reward: {}\n\n'.format (reward)
        self.log_info.write (self.info)
        print (self.info)
        return self.observation, reward, done, self.info
