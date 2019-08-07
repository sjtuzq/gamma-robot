#!/usr/bin/env python3

"""
action 49: plugging something into something
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


class Engine49(Engine):
    def __init__(self, opt, worker_id=None):
        super(Engine49,self).__init__(opt)
        self.opt = opt 
        self._wid = worker_id
        self.physical_id = opt.p
        self.robot.gripperMaxForce = 1000.0
        self.robot.armMaxForce = 500.0

        self.opt.p.setPhysicsEngineParameter (constraintSolverType=self.opt.p.CONSTRAINT_SOLVER_LCP_DANTZIG,
                                             globalCFM=0.000001)

    def init_obj(self):
        self.obj_position = [0.42, -0.06, 0.3315]
        self.obj_orientation = [0, 0, -0.1494381, 0.9887711]
        self.obj_scaling = 1 / 6.26 * 3.0
        self.obj_id = self.p.loadURDF( os.path.join(self.env_root, "urdf/obj_libs/bottles/b4/b4.urdf"),basePosition=self.obj_position,baseOrientation=self.obj_orientation,globalScaling=self.obj_scaling)
        print("self.p.getAABB of peg")
        AABB= self.p.getAABB(self.obj_id)
        print(AABB[1][0]-AABB[0][0])

        self.box_w = 0.30
        self.box_h = 0.0
        self.box_d = 0.30
        
        self.hole_r = 0.02
        mass = 0

        obj1_w = 0.05
        obj1_h = 0.02
        obj1_d = 0.025
        self.obj1_position = [self.box_w, self.box_h - self.hole_r - obj1_h, self.box_d]
        self.obj1_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj1_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj1_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj1_id = self.p.createMultiBody(mass, self.obj1_c, self.obj1_v, self.obj1_position)
        self.p.changeVisualShape (self.obj1_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])
 
        self.obj2_position = [self.box_w, self.box_h + self.hole_r + obj1_h, self.box_d]
        self.obj2_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj2_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj2_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj1_w, obj1_h, obj1_d])
        self.obj2_id = self.p.createMultiBody(mass, self.obj2_c, self.obj2_v, self.obj2_position)
        self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])
 
        obj3_w = 0.015
        obj3_h = 0.02
        obj3_d = 0.025
        self.obj3_position = [self.box_w - self.hole_r - obj3_w, self.box_h, self.box_d]
        self.obj3_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj3_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj3_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj3_id = self.p.createMultiBody(mass, self.obj3_c, self.obj3_v, self.obj3_position)
        self.p.changeVisualShape (self.obj3_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])
 
        self.obj4_position = [self.box_w + self.hole_r + obj3_w, self.box_h, self.box_d]
        self.obj4_orientation = [0.0, 0.0, 0.0, 1.0]
        self.obj4_v = self.p.createVisualShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj4_c = self.p.createCollisionShape(self.p.GEOM_BOX,halfExtents=[obj3_w, obj3_h, obj3_d])
        self.obj4_id = self.p.createMultiBody(mass, self.obj4_c, self.obj4_v, self.obj4_position)
        self.p.changeVisualShape (self.obj4_id, -1, rgbaColor=[1.,0.,0.,1],specularColor=[1.,1.,1.])
 
        #self.p.changeVisualShape (self.obj2_id, -1, rgbaColor=[1.,1.,0, 1.])


    def reset_obj(self):
        self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)
        self.p.resetBasePositionAndOrientation(self.obj2_id,self.obj2_position,self.obj2_orientation)
        
        obj_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj_id, -1, lateralFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, rollingFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, spinningFriction=obj_friction_ceof)
        self.p.changeDynamics(self.obj_id, -1, linearDamping=40.0)
        self.p.changeDynamics(self.obj_id, -1, angularDamping=1.0)
        self.p.changeDynamics(self.obj_id, -1, contactStiffness=1.0, contactDamping=0.9)


        obj2_friction_ceof = 2000.0
        self.p.changeDynamics(self.obj2_id, -1, mass=0.05)
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
            if i > 180:
              glist[i] = 230
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
