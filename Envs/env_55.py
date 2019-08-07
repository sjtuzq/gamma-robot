#!/usr/bin/env python3

"""
action 55: poking something so it slightly moves
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

try:
    from .env import Engine
    from .utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
except Exception:
    from env import Engine
    from utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code

class Engine55(Engine):
    def __init__(self, opt, worker_id=None):
        super(Engine55,self).__init__(opt)
        self.opt = opt 
        self._wid = worker_id
        self.physical_id = opt.p
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14

        self.opt.p.setPhysicsEngineParameter (constraintSolverType=self.opt.p.CONSTRAINT_SOLVER_LCP_DANTZIG,
                                             globalCFM=0.000001)


    def init_obj(self):
        self.obj_position = [0.42, -0.08, 0.3]
        self.obj_scaling = 1.0
        self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.obj_id = self.p.loadURDF(os.path.join(self.env_root,"urdf/obj_libs/bottles/b3/b3.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)

        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[1.,0.,0.,1])

    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
        
        obj_friction_ceof = 0.3
        self.p.changeDynamics(self.obj_id, -1, mass=0.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)



    def init_grasp(self):
        self.robot.gripperControl(255)    
  
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

        self.start_pos = p.getLinkState (self.robotId, 7)[0]


    def get_handcraft_reward (self):
        distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5

        obj = p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        aabb_dist = sum ([x ** 2 for x in obj_center]) ** 0.5

        reward = (0.5-aabb_dist)*10000

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
                reward = -10000

        if aabb_dist<0.1:
            done = True
            reward = 10000

        # reward = -1
        self.info += 'reward: {}\n\n'.format (reward)
        self.log_info.write (self.info)
        print (self.info)
        return self.observation, reward, done, 1#self.info
