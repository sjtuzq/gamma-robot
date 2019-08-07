#!/usr/bin/env python3

"""
action 46: opening something
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

class Engine46(Engine):
    def __init__(self, opt, worker_id=None):
        super(Engine46,self).__init__(opt)
        self.opt = opt 
        self._wid = worker_id
        self.physical_id = opt.p
        self.robot.gripperMaxForce = 200.0
        self.robot.armMaxForce = 200.0
        self.robot.jd = [0.01] * 14
        self.opt.p.setPhysicsEngineParameter (constraintSolverType=self.opt.p.CONSTRAINT_SOLVER_LCP_DANTZIG,
                                             globalCFM=0.000001)


    def reset_new(self):
        print("reset in env15")
        self.physical_id = self.p

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.8)
        self.p.setTimeStep (1/250.)
        try:
            self.epoch_num += 1
        except:
            self.epoch_num = 0

        self.log_path = safe_path(os.path.join(self.log_root,'epoch-{}'.format(self.epoch_num)))
        self.log_info = open(os.path.join(self.log_root,'epoch-{}.txt'.format(self.epoch_num)),'w')
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()
        self.init_grasp ()
      
        return self.get_observation()

    def init_obj(self):
        self.obj_id = self.p.loadURDF( os.path.join(self.env_root, "urdf/obj_libs/bottles/b5/b5.urdf"),useFixedBase=True)
        self.obj2_id = self.p.loadURDF( os.path.join(self.env_root, "urdf/obj_libs/bottle_caps/c1/c1.urdf"))
        self.p.changeVisualShape (self.obj_id, -1, rgbaColor=[0.2,0.,0.8,1])
        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1.,0.,0.,1])
 
   
    def reset_obj(self):

        self.obj_x = 0.32
        self.obj_y = 0.0
        self.obj_z = 0.42

        self.obj1_ori =  p.getQuaternionFromEuler ([0,math.pi/2.0, 0 ])
        self.p.resetBasePositionAndOrientation(self.obj_id,[self.obj_x ,  self.obj_y, self.obj_z],self.obj1_ori)
        self.obj2_ori =  p.getQuaternionFromEuler ([math.pi/2.0,0,0])
        self.p.resetBasePositionAndOrientation(self.obj2_id,[self.obj_x , self.obj_y, self.obj_z + 0.08 ],self.obj2_ori)
  
        obj_friction_ceof = 0.3
        self.p.changeDynamics(self.obj_id, -1, mass=0.9)
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)

        obj2_friction_ceof = 0.4
        self.p.changeDynamics(self.obj2_id, -1, mass=0.9)
        self.p.changeDynamics(self.obj2_id, -1, lateralFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, rollingFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, spinningFriction=obj2_friction_ceof)
        self.p.changeDynamics(self.obj2_id, -1, linearDamping=20.0)
        self.p.changeDynamics(self.obj2_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj2_id, -1, contactStiffness=1.0, contactDamping=0.9)

        table_friction_ceof = 0.4
        self.p.changeDynamics(self.table_id, -1, lateralFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, rollingFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, spinningFriction=table_friction_ceof)
        self.p.changeDynamics(self.table_id, -1, contactStiffness=1.0, contactDamping=0.9)

    def init_motion(self):
        self.data_q = np.load (os.path.join(self.env_root,"init/47_4_q.npy"))
        self.data_gripper = np.load (self.env_root + '/init/gripper.npy')
        self.robot.setJointValue(self.data_q[0],gripper=self.data_gripper[0])

    def init_grasp(self):
        self.robot.gripperControl(0)

        qlist = np.load( os.path.join(self.env_root,"init/47_4_q.npy"))
        glist = np.load( os.path.join(self.env_root,"init/47_4_gripper.npy"))
        num_q = len(qlist[0])

        for i in range(40,len(qlist),1):
            glist[i] = min(130,glist[i])
            self.robot.jointPositionControl(qlist[i],gripper=glist[i])
 
        self.fix_orn = p.getLinkState(self.robotId, 7)[1]
        self.fix_orn = [self.fix_orn]
        self.start_pos = p.getLinkState (self.robotId, 7)[0]


    def get_handcraft_reward (self):
        distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5

        obj2 = p.getAABB (self.obj2_id, -1)
        obj2_center = [(x + y) * 0.5 for x, y in zip (obj2[0], obj2[1])]
        obj = p.getAABB (self.obj_id, -1)
        obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
        aabb_dist = sum ([(x - y) ** 2 for x, y in zip (obj2_center, obj_center)]) ** 0.5

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
