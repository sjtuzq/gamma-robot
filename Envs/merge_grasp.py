
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


def init_grasp_8 (self):
    try:
        p.removeBody (self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState (self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl (0)

    for init_t in range (100):
        box = p.getAABB (self.obj_id, -1)
        center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        center[0] -= 0.05
        center[1] -= 0.05
        center[2] += 0.03
        # center = (box[0]+box[1])*0.5
    points = np.array ([pos_traj[0], center])

    start_id = 0
    init_traj = point2traj (points)
    start_id = self.move (init_traj, orn_traj, start_id)

    # grasping
    grasp_stage_num = 10
    for grasp_t in range (grasp_stage_num):
        gripperPos = grasp_t / float (grasp_stage_num) * 180.0
        self.robot.gripperControl (gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj ([pos, [pos[0] + 0.1, pos[1] + 0.08, pos[2] + 0.25]])
    start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1], pos[2] + (random.random () - 0.5) * 0.1]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1] + (random.random () - 0.5) * 0.2 + 0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0] + (random.random () - 0.5) * 0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random ()
        if prob < 0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.45 + (random.random () - 0.5) * 0.2, -0.1 + (random.random () - 0.5) * 0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)  # , physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.38, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi / 2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)  # , physicsClientId=self.physical_id)

    texture_path = os.path.join (self.env_root, 'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_9(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]-0.2, pos[1]+0.12, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.obj_id, -1, rgbaColor=[2, 0, 0, 1])

    # self.p.changeDynamics (self.obj_id, -1, linearDamping=20.0)
    # self.p.changeDynamics (self.obj_id, -1, angularDamping=10.0)
    # self.p.changeDynamics (self.obj_id, -1, contactStiffness=1, contactDamping=0.1)

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_10(self):
    try:
        p.removeBody (self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState (self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl (0)

    for init_t in range (100):
        box = p.getAABB (self.obj_id, -1)
        center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
        center[0] -= 0.05
        center[1] -= 0.05
        center[2] += 0.03
        # center = (box[0]+box[1])*0.5
    points = np.array ([pos_traj[0], center])

    start_id = 0
    init_traj = point2traj (points)
    start_id = self.move (init_traj, orn_traj, start_id)

    # grasping
    grasp_stage_num = 10
    for grasp_t in range (grasp_stage_num):
        gripperPos = grasp_t / float (grasp_stage_num) * 180.0
        self.robot.gripperControl (gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.08, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_11(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]-0.08, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_12(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]+0.03, pos[1]+0.13, pos[2]+0.3]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.28]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1



def init_grasp_16(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.13, pos[2]+0.2]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_17(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]+0.1, pos[1]+0.08, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)


    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.33, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_18(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    # up_traj = point2traj([pos, [pos[0]-0.2, pos[1]+0.12, pos[2]+0.05]])
    up_traj = point2traj([pos, [pos[0]-0.2, pos[1]+0.12, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_19(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]-0.08, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_20(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]+0.03, pos[1]+0.13, pos[2]+0.3]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.48, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, math.pi, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])


    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1



def init_grasp_40(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

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
    up_traj = point2traj([pos, [pos[0], pos[1]+0.1, pos[2]+0.1]])
    start_id = self.move(up_traj, orn_traj,start_id)


    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
        self.box_position = [0.42, 0.16, 0.33]
        self.box_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.box_scaling = 0.15
        self.box_id = self.p.loadURDF(fileName=self.box_file, basePosition=self.box_position,baseOrientation=self.box_orientation,
                                 globalScaling=self.box_scaling)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])

    # texture_path = os.path.join(self.env_root,'texture/sun_textures')
    # texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    # textid = p.loadTexture (texture_file)
    # # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # box = p.getAABB (self.box_id, -1)
    # box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    # obj = p.getAABB (self.obj_id, -1)
    # obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    # self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

    def init_grasp(self):
        pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
        orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
        self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        self.robot.gripperControl(0)

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

        # grasping
        grasp_stage_num = 10
        for grasp_t in range(grasp_stage_num):
            gripperPos = grasp_t / float(grasp_stage_num) * 180.0
            self.robot.gripperControl(gripperPos)
            start_id += 1

        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0]-0.15, pos[1]+0.13, pos[2]+0.25]])
        start_id = self.move(up_traj, orn_traj,start_id)

        self.start_pos = p.getLinkState (self.robotId, 7)[0]

        if self.opt.rand_box == 'rand':
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.3, pos[1] + (random.random()-0.5)*0.3, pos[2] + (random.random()-0.5)*0.3]])
            start_id = self.move (up_traj, orn_traj, start_id)

        self.start_pos = p.getLinkState (self.robotId, 7)[0]

def init_grasp_41(self):
    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]-0.15, pos[1]+0.13, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    if self.opt.rand_box == 'rand':
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.3, pos[1] + (random.random()-0.5)*0.3, pos[2] + (random.random()-0.5)*0.3]])
        start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

def init_grasp_42(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

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
    up_traj = point2traj([pos, [pos[0], pos[1]+0.04, pos[2]+0.15]])
    start_id = self.move(up_traj, orn_traj,start_id)


    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
        self.box_position = [0.42, 0.16, 0.33]
        self.box_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.box_scaling = 0.15
        self.box_id = self.p.loadURDF(fileName=self.box_file, basePosition=self.box_position,baseOrientation=self.box_orientation,
                                 globalScaling=self.box_scaling)

    # texture_path = os.path.join(self.env_root,'texture/sun_textures')
    # texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    # textid = p.loadTexture (texture_file)
    # # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # box = p.getAABB (self.box_id, -1)
    # box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    # obj = p.getAABB (self.obj_id, -1)
    # obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    # self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_43(self):

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
    # up_traj = point2traj([pos, [pos[0]-0.1, pos[1]+0.08, pos[2]+0.3]])
    up_traj = point2traj([pos, [pos[0]-0.1, pos[1]+0.08, pos[2]+0.2]])
    start_id = self.move(up_traj, orn_traj,start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

def init_grasp_44(self):
    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]-0.15, pos[1]+0.13, pos[2]+0.25]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_box == 'rand':
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.3, pos[1] + (random.random()-0.5)*0.3, pos[2] + (random.random()-0.5)*0.3]])
        start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

def init_grasp_45(self):
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



def init_grasp_85(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0]+0.05, pos[1]+0.25, pos[2]]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.30, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    # p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 1, 1, 0.9])
    # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_86(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.2, pos[2]]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_87(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    self.data_q = np.load (self.env_root + '/init/left_q.npy')
    self.data_dq = np.load (self.env_root + '/init/left_dq.npy')
    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

    orn_traj = np.array([p.getLinkState (self.robotId, 7)[1]]*orn_traj.shape[0])
    self.fix_orn = np.array([p.getLinkState (self.robotId, 7)[1]]*orn_traj.shape[0])

    self.obj_position = [0.4, -0.08, 0.34]
    self.obj_scaling = 2
    self.obj_orientation = self.p.getQuaternionFromEuler ([math.pi / 2, -math.pi / 2, 0])
    self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)


    for init_t in range(100):
        box = p.getAABB(self.obj_id,-1)
        center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
        center[0] -= 0.03
        center[1] += 0.02
        center[2] += 0.05
        # center = (box[0]+box[1])*0.5
    points = np.array ([pos_traj[0], center])

    start_id = 0
    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, center])
    start_id = self.move(up_traj, orn_traj,start_id)

    # init_traj = point2traj(points)
    # start_id = self.move(init_traj,orn_traj,start_id)

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 220.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.2, pos[2]+0.06]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

def init_grasp_93(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    self.data_q = np.load (self.env_root + '/init/left_q.npy')
    self.data_dq = np.load (self.env_root + '/init/left_dq.npy')
    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

    orn_traj = np.array([p.getLinkState (self.robotId, 7)[1]]*orn_traj.shape[0])
    self.fix_orn = np.array([p.getLinkState (self.robotId, 7)[1]]*orn_traj.shape[0])

    self.obj_position = [0.4, -0.08, 0.34]
    self.obj_scaling = 2
    self.obj_orientation = self.p.getQuaternionFromEuler ([math.pi / 2, -math.pi / 2, 0])
    self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)


    for init_t in range(100):
        box = p.getAABB(self.obj_id,-1)
        center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
        center[0] -= 0.03
        center[1] += 0.02
        center[2] += 0.05
        # center = (box[0]+box[1])*0.5
    points = np.array ([pos_traj[0], center])

    start_id = 0
    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, center])
    start_id = self.move(up_traj, orn_traj,start_id)

    # init_traj = point2traj(points)
    # start_id = self.move(init_traj,orn_traj,start_id)

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 220.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.2, pos[2]+0.06]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # grasp_stage_num = 100
    # for grasp_t in range (grasp_stage_num):
    #     gripperPos = (grasp_stage_num-grasp_t) / float (grasp_stage_num) * 180.0
    #     self.robot.gripperControl (gripperPos)
    #     start_id += 1

def init_grasp_94(self):
    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
    self.robot.gripperControl(0)

    self.obj_position = [0.42, -0.08, 0.34]
    self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
    self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
    p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    # for init_t in range(100):
    #     box = p.getAABB(self.obj_id,-1)
    #     center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
    #     center[0] -= 0.05
    #     center[1] -= 0.05
    #     center[2] += 0.03
    #     # center = (box[0]+box[1])*0.5
    # points = np.array ([pos_traj[0], center])
    #
    # start_id = 0
    # init_traj = point2traj(points)
    # start_id = self.move(init_traj,orn_traj,start_id)

    start_id = 0
    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.2, pos[2]-0.04]])
    start_id = self.move(up_traj, orn_traj, start_id)


    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    self.start_pos = p.getLinkState (self.robotId, 7)[0]




def init_grasp_100(self):
    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
    self.robot.gripperControl(0)

    self.obj_position = [0.42, -0.08, 0.30]
    self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
    self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
    p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    # for init_t in range(100):
    #     box = p.getAABB(self.obj_id,-1)
    #     center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
    #     center[0] -= 0.05
    #     center[1] -= 0.05
    #     center[2] += 0.03
    #     # center = (box[0]+box[1])*0.5
    # points = np.array ([pos_traj[0], center])
    #
    # start_id = 0
    # init_traj = point2traj(points)
    # start_id = self.move(init_traj,orn_traj,start_id)

    start_id = 0
    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.15, pos[2]-0.04]])
    start_id = self.move(up_traj, orn_traj, start_id)


    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_box == 'rand':
        self.obj_position = [0.42+(random.random()-0.5)*0.2, -0.08+(random.random()-0.1)*0.2, 0.30]
        self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
        self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)

    # texture_path = os.path.join(self.env_root,'texture/sun_textures')
    # texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    # textid = p.loadTexture (texture_file)
    # # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    # p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])

    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # box = p.getAABB (self.box_id, -1)
    # box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    # obj = p.getAABB (self.obj_id, -1)
    # obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    # self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_101(self):
    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])
    self.robot.gripperControl(0)

    self.obj_position = [0.42, -0.08, 0.34]
    self.obj_orientation = self.p.getQuaternionFromEuler ([-math.pi / 2, 0, 0])
    self.p.resetBasePositionAndOrientation (self.obj_id, self.obj_position, self.obj_orientation)
    p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])

    # for init_t in range(100):
    #     box = p.getAABB(self.obj_id,-1)
    #     center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
    #     center[0] -= 0.05
    #     center[1] -= 0.05
    #     center[2] += 0.03
    #     # center = (box[0]+box[1])*0.5
    # points = np.array ([pos_traj[0], center])
    #
    # start_id = 0
    # init_traj = point2traj(points)
    # start_id = self.move(init_traj,orn_traj,start_id)

    start_id = 0
    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.25, pos[2]-0.04]])
    start_id = self.move(up_traj, orn_traj, start_id)


    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
        self.box_position = [0.42+(random.random()-0.5)*0.2, 0.03+(random.random()-0.5)*0.4, 0.30]
        self.box_scaling = 0.15
        self.box_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        try:
            self.p.resetBasePositionAndOrientation (self.box_id, self.box_position, self.box_orientation)
        except:
            self.box_id = self.p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                           baseOrientation=self.box_orientation,
                                           globalScaling=self.box_scaling)
    else:
        self.box_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
        self.box_position = [0.42, 0.13, 0.33]
        self.box_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.box_scaling = 0.15
        try:
            self.p.resetBasePositionAndOrientation (self.box_id, self.box_position, self.box_orientation)
        except:
            self.box_id = self.p.loadURDF(fileName=self.box_file, basePosition=self.box_position,baseOrientation=self.box_orientation,
                                 globalScaling=self.box_scaling)

    # texture_path = os.path.join(self.env_root,'texture/sun_textures')
    # texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    # textid = p.loadTexture (texture_file)
    # # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    # box = p.getAABB (self.box_id, -1)
    # box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    # obj = p.getAABB (self.obj_id, -1)
    # obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    # self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_104(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.05, pos[2]+0.3]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.42+(random.random()-0.5)*0.2, 0.00+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.42, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    # p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 1, 1, 0.9])
    # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_105(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.05, pos[2]+0.3]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.42+(random.random()-0.5)*0.2, 0.00+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.42, 0.00, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, math.pi/2])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    # p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 1, 1, 0.9])
    # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_106(self):
    pos_traj = np.load (os.path.join (self.env_root, 'init', 'pos.npy'))
    orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
    self.fix_orn = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))

    for j in range (7):
        self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

    self.robot.gripperControl(0)

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

    # grasping
    grasp_stage_num = 10
    for grasp_t in range(grasp_stage_num):
        gripperPos = grasp_t / float(grasp_stage_num) * 180.0
        self.robot.gripperControl(gripperPos)
        start_id += 1

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj([pos, [pos[0], pos[1]+0.05, pos[2]+0.3]])
    start_id = self.move(up_traj, orn_traj,start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2+0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/openbox/openbox.urdf")
        self.box_position = [0.45+(random.random()-0.5)*0.2, -0.1+(random.random()-0.5)*0.4, 0.34]
        self.box_scaling = 0.0003
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join (self.env_root, "urdf/openbox/openbox.urdf")
        self.box_position = [0.45, 0.05, 0.34]
        self.box_scaling = 0.00037
        self.box_orientation = p.getQuaternionFromEuler ([0, 0, 0])
        self.box_id = p.loadURDF (fileName=self.box_file, basePosition=self.box_position,
                                  baseOrientation=self.box_orientation,
                                  globalScaling=self.box_scaling)#, physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    # p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 1, 1, 0.9])
    # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5

def init_grasp_107(self):
    try:
        p.removeBody(self.box_id)
    except:
        pass

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
    up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+0.3]])
    start_id = self.move(up_traj, orn_traj,start_id)

    traj_path = os.path.join (self.opt.project_root, 'scripts', 'Dmp', 'traj.npy')
    t_pos = np.load(traj_path)[0]
    up_traj = point2traj ([pos, t_pos])
    start_id = self.move (up_traj, orn_traj, start_id)

    pos = p.getLinkState (self.robotId, 7)[0]
    up_traj = point2traj ([pos, [pos[0], pos[1]+0.15, pos[2]]])
    start_id = self.move (up_traj, orn_traj, start_id)

    if self.opt.rand_start == 'rand':
        # move in z-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj([pos, [pos[0], pos[1], pos[2]+(random.random()-0.5)*0.1]])
        start_id = self.move(up_traj, orn_traj,start_id)

        # move in y-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1]+(random.random()-0.5)*0.2, pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

        # move in x-axis direction
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0]+(random.random()-0.5)*0.2, pos[1], pos[2]]])
        start_id = self.move (up_traj, orn_traj, start_id)

    elif self.opt.rand_start == 'two':
        prob = random.random()
        if prob<0.5:
            pos = p.getLinkState (self.robotId, 7)[0]
            up_traj = point2traj ([pos, [pos[0], pos[1] + 0.2, pos[2]]])
            start_id = self.move (up_traj, orn_traj, start_id)


    if self.opt.rand_box == 'rand':
        self.box_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
        self.box_position = [0.37+(random.random()-0.5)*0.2, 0.03+(random.random()-0.5)*0.3, 0.34]
        self.box_orientation = p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.box_scaling = 0.21
        self.box_id = p.loadURDF(fileName=self.box_file, basePosition=self.box_position,baseOrientation=self.box_orientation,
                                 globalScaling=self.box_scaling)#physicsClientId=self.physical_id)
    else:
        self.box_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
        self.box_position = [0.37, 0.03, 0.34]
        self.box_position = [0.37, 0.18, 0.34]
        self.box_orientation = p.getQuaternionFromEuler([-math.pi/2, 0, 0])
        self.box_scaling = 0.21
        self.box_id = p.loadURDF(fileName=self.box_file, basePosition=self.box_position,baseOrientation=self.box_orientation,
                                 globalScaling=self.box_scaling)#physicsClientId=self.physical_id)

    texture_path = os.path.join(self.env_root,'texture/sun_textures')
    texture_file = os.path.join (texture_path, random.sample (os.listdir (texture_path), 1)[0])
    textid = p.loadTexture (texture_file)
    p.changeVisualShape (self.box_id, -1, rgbaColor=[1, 0, 0, 1])
    # p.changeVisualShape (self.box_id, -1, textureUniqueId=textid)
    self.start_pos = p.getLinkState (self.robotId, 7)[0]

    box = p.getAABB (self.box_id, -1)
    box_center = [(x + y) * 0.5 for x, y in zip (box[0], box[1])]
    obj = p.getAABB (self.obj_id, -1)
    obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
    self.last_aabb_dist = sum ([(x - y) ** 2 for x, y in zip (box_center, obj_center)]) ** 0.5
    self.last_aabb_dist_storage = [self.last_aabb_dist]*20

