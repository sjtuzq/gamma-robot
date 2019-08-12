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
"""
try:
    from .utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
    from .robot import Robot
except Exception:
    from utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
    from robot import Robot
"""
if __name__ == "__main__":
    from .utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
    from .robot import Robot
else:
    from utils import get_view,safe_path,cut_frame,point2traj,get_gripper_pos,backup_code
    from robot import Robot

import pkgutil
egl = pkgutil.get_loader ('eglRenderer')

class Engine:
    def __init__(self,opt):
        self.opt = opt
        self.p = opt.p
        self.class_id = opt.action_id
        self.video_id = opt.video_id
        self.test_id = opt.test_id
        self.cut_frame_num = opt.cut_frame_num
        self.give_reward_num = opt.give_reward_num
        self.init_rl ()

        if self.opt.video_reward:
            self.eval = self.opt.load_video_pred

        if self.opt.use_cycle:
            self.cycle = self.opt.load_cycle

        if self.opt.use_dmp:
            self.dmp = self.opt.load_dmp
            # assert (self.opt.video_reward)

        self.dataset_root = os.path.join(opt.project_root,'dataset')
        self.log_root = os.path.join(opt.project_root,'logs')
        self.log_root = safe_path(self.log_root+'/td3_log/test{}'.format(self.test_id))

        self.env_root = os.path.join(opt.project_root,'scripts','Envs')
        self.script_root = os.path.join(opt.project_root,'scripts')
        self.embedding_data = np.load(os.path.join(self.script_root,'utils','labels','label_embedding_uncased.npy'))
        self.memory_path = safe_path(os.path.join(self.log_root,'memory'))
        backup_code (self.script_root, self.log_root)

        self.init_plane ()
        self.init_table ()
        self.robot = Robot(pybullet_api=self.p,opt=self.opt)
        self.robotId = self.robot.robotId
        self.robotEndEffectorIndex = self.robot.endEffectorIndex
        self.init_obj ()

        self.view_matrix, self.proj_matrix = get_view (self.opt)
        self.q_home = np.array((0., -np.pi/6., 0., -5./6.*np.pi, 0., 2./3.*np.pi, 0.))
        self.w = self.opt.img_w
        self.h = self.opt.img_h


    def destroy(self):
        p.disconnect(self.physical_id)

    def init_table(self):
        table_path = os.path.join(self.env_root,'urdf/table/table.urdf')
        self.table_id = p.loadURDF(table_path, [0.42,0,0],[0,0,math.pi*0.32,1],globalScaling=0.6)#,physicsClientId=self.physical_id)
        texture_path = os.path.join(self.env_root,'texture/table_textures/table_texture.jpg')
        table_textid = p.loadTexture (texture_path)
        p.changeVisualShape (self.table_id, -1, textureUniqueId=table_textid)

    def init_motion(self):
        self.data_q = np.load (self.env_root + '/init/q.npy')
        self.data_dq = np.load (self.env_root + '/init/dq.npy')
        self.data_gripper = np.load (self.env_root + '/init/gripper.npy')

    def init_running (self):
        self.motion_path = self.dataset_root+'/actions/{}-{}'.format(self.class_id,self.video_id)
        self.data_q = np.load (self.dataset_root+'/actions/{}-{}/q.npy'.format(self.class_id,self.video_id))
        self.data_dq = np.load(self.dataset_root+'/actions/{}-{}/dq.npy'.format(self.class_id,self.video_id))
        self.data_gripper = np.load(self.dataset_root+'/actions/{}-{}/gripper.npy'.format(self.class_id,self.video_id))

        self.frames_path = cut_frame(self.dataset_root+'/actions/{}-{}/{}-{}.avi'.format (self.class_id, self.video_id,self.class_id,self.video_id),
                                     self.dataset_root+'/actions/{}-{}/frames'.format (self.class_id, self.video_id))
        self.output_path = safe_path(self.dataset_root+'/actions/{}-{}/simulation_frames'.format(self.class_id,self.video_id))
        self.mask_frames_path = safe_path(self.dataset_root+'/actions/{}-{}/masked_frames'.format(self.class_id,self.video_id))

    def init_plane(self):
        self.plane_id = self.p.loadURDF (os.path.join(self.env_root,'urdf/table/plane.urdf'), [0.7, 0, 0], [0, 0, -math.pi * 0.02, 1], globalScaling=0.7)
        texture_path = os.path.join(self.opt.project_root,'scripts','Envs','texture/real_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        self.textid = p.loadTexture(texture_file)
        self.p.changeVisualShape (self.plane_id, -1, rgbaColor=[1, 1, 1, 0.9])
        self.p.changeVisualShape (self.plane_id, -1, textureUniqueId=self.textid)


    def save_video(self,img_info,i):
        img = img_info[2][:, :, :3]
        mask = (img_info[4] > 10000000)
        mask_id_label = [234881025, 301989889, 285212673, 268435457, 318767105, 335544321, 201326593, 218103809, 167772161]
        for item in mask_id_label:
            mask = mask * (img_info[4] != item)
        img = cv2.cvtColor (img, cv2.COLOR_RGB2BGR)
        img[mask] = [127, 151, 182]
        cv2.imwrite (os.path.join (self.output_path, '%06d.jpg' % (i)), img)

        try:
            img = cv2.imread (os.path.join (self.frames_path, '%06d.jpg' % (i + 1)))
            img[mask] = [127, 151, 182]
            cv2.imwrite (os.path.join (self.mask_frames_path, '%06d.jpg' % (i)), img)
        except:
            print('no video frame:{}'.format(i))

    def init_obj(self):
        if self.opt.object_id == 'bottle':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/bottle1.urdf")
            self.obj_position = [0.4, -0.15, 0.42]
            self.obj_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
            self.obj_scaling = 1.4
            self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)

        if self.opt.object_id == 'cup':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
            self.obj_position = [0.45, -0.18, 0.34]
            self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
            self.obj_scaling = 0.11
            self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)

        if self.opt.object_id == 'nut':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/nut.urdf")
            self.obj_position = [0.4, -0.15, 0.34]
            self.obj_scaling = 2
            self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
            self.obj_id = self.p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling)#,physicsClientId=self.physical_id)

        texture_path = os.path.join(self.env_root,'texture/sun_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        textid = self.p.loadTexture(texture_file)
        # self.p.changeVisualShape (self.obj_id, -1, textureUniqueId=textid)

    def reset_obj(self):
        if self.opt.object_id == 'bottle':
            self.obj_position = [0.4, -0.15, 0.42]
            self.obj_orientation = p.getQuaternionFromEuler([math.pi/2, 0, 0])
            self.obj_scaling = 1.4
            self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)#,physicsClientId=self.physical_id)

        if self.opt.object_id == 'cup':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
            self.obj_position = [0.45, -0.18, 0.34]
            self.obj_orientation = self.p.getQuaternionFromEuler([-math.pi/2, 0, 0])
            self.obj_scaling = 0.11
            self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

        if self.opt.object_id == 'nut':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/nut.urdf")
            self.obj_position = [0.4, -0.15, 0.34]
            self.obj_scaling = 2
            self.obj_orientation = self.p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
            self.p.resetBasePositionAndOrientation(self.obj_id,self.obj_position,self.obj_orientation)

        texture_path = os.path.join(self.env_root,'texture/sun_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        textid = self.p.loadTexture(texture_file)
        # self.p.changeVisualShape (self.obj_id, -1, textureUniqueId=textid)
        p.changeVisualShape (self.obj_id, -1, rgbaColor=[1, 0, 0, 1])



    def run(self):
        for i in range(self.data_q.shape[0]):
            jointPoses = self.data_q[i]
            for j in range(self.robotEndEffectorIndex):
                p.resetJointState(self.robotId, j, jointPoses[j], self.data_dq[i][j])

            gripper = self.data_gripper[i]
            self.gripperOpen = 1 - gripper / 255.0
            self.gripperPos = np.array (self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array (
                self.gripperLowerLimitList) * self.gripperOpen
            for j in range (6):
                index_ = self.activeGripperJointIndexList[j]
                p.resetJointState (self.robotId, index_, self.gripperPos[j], 0)

            img_info = p.getCameraImage (width=self.w,
                                         height=self.h,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=-1,
                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                         renderer=p.ER_TINY_RENDERER)
            self.save_video(img_info,i)
            # time.sleep (0.01)
            p.stepSimulation()

    def get_traj(self):
        pos_traj, orn_traj = [], []
        for i in range (self.data_q.shape[0]):
            poses = self.data_q[i]
            for j in range (7):
                p.resetJointState (self.robotId, j, poses[j], self.data_dq[i][j])

            state = p.getLinkState (self.robotId, 7)
            pos = state[0]
            orn = state[1]
            pos_traj.append (pos)
            orn_traj.append (orn)
            print(i)

        # np.save (os.path.join(self.env_root,'init','pos.npy'), np.array (pos_traj))
        # np.save (os.path.join(self.env_root,'init','orn.npy'), np.array (orn_traj))

    def init_grasp(self):
        pos_traj = np.load (os.path.join(self.env_root,'init','pos.npy'))
        orn_traj = np.load (os.path.join(self.env_root,'init','orn.npy'))
        self.fix_orn = np.load (os.path.join(self.env_root,'init','orn.npy'))

        for j in range (7):
            self.p.resetJointState(self.robotId, j, self.data_q[0][j], self.data_dq[0][j])

        for init_t in range(100):
            box = self.p.getAABB(self.obj_id,-1)
            center = [(x+y)*0.5 for x,y in zip(box[0],box[1])]
            center[0] -= 0.05
            center[1] -= 0.05
            center[2] += 0.03
            # center = (box[0]+box[1])*0.5
        points = np.array ([pos_traj[0], center])

        start_id = 0
        init_traj = point2traj(points)
        start_id = self.move(init_traj,orn_traj,start_id)

        self.p.stepSimulation()

        # grasping
        grasp_stage_num = 10
        for grasp_t in range(grasp_stage_num):
            gripperPos = (grasp_t + 1.)/ float(grasp_stage_num) * 250.0 + 0.0
            self.robot.gripperControl(gripperPos)

            start_id += 1

        pos = p.getLinkState (self.robotId, 7)[0]
        left_traj = point2traj([pos, [pos[0], pos[1]+0.14, pos[2]+0.05]])
        start_id = self.move(left_traj, orn_traj,start_id)

        self.start_pos = p.getLinkState(self.robotId,7)[0]

    def move_up(self):
        # move in z-axis direction
        orn_traj = np.load (os.path.join(self.env_root,'init','orn.npy'))
        pos = p.getLinkState (self.robotId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1], pos[2] + 0.3]],delta=0.005)
        start_id = self.move (up_traj, orn_traj,0)

    def explore(self,traj):
        orn_traj = np.load ('orn.npy')
        start_id = self.move (traj, orn_traj,0)

    def move(self,pos_traj,orn_traj,start_id=0):
        for i in range(int(len(pos_traj))):
            pos = pos_traj[i]
            orn = orn_traj[i]
            self.robot.operationSpacePositionControl(pos=pos,orn=orn,null_pose=self.data_q[i])
            # self.robot.operationSpacePositionControl(pos=pos,orn=orn)

            img_info = p.getCameraImage (width=self.w,
                                         height=self.h,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=-1,
                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                         renderer=p.ER_TINY_RENDERER)
            # self.save_video(img_info,start_id+i)
        return start_id+len(pos_traj)

    def init_rl(self):
        self.target_pos = [0.3633281737186908, -0.23858468351424078, 0.670415682662147]
        # self.target_pos = [0.3633281737186908, -0.23858468351424078+0.1, 0.670415682662147-0.1]
        if self.opt.observation == 'joint_pose':
            self.observation_space = 7
        elif self.opt.observation == 'end_pos':
            self.observation_space = 3
        elif self.opt.observation == 'before_cnn':
            self.observation_space = 256
        else:
            self.observation_space = 32
        self.each_action_lim = self.opt.each_action_lim
        if self.opt.add_gripper:
            low = [-self.each_action_lim] * 4
            high = [self.each_action_lim] * 4
        else:
            low = [-self.each_action_lim] * 3
            high = [self.each_action_lim] * 3
        self.action_space = {'low':low,'high':high}
        self.max_seq_num = 500
        self.min_dis_lim = self.opt.end_distance
        self.axis_limit = [eval(self.opt.axis_limit_x),eval(self.opt.axis_limit_y),
                           eval(self.opt.axis_limit_z)]

    def init_dmp(self):
        if self.opt.use_dmp and self.opt.dmp_imitation:
            trajectories = []
            for file in os.listdir (self.opt.actions_root):
                action_id = int (file.split ('-')[0])
                video_id = int (file.split('-')[1])
                if (action_id == self.opt.action_id) and (video_id==self.opt.video_id):
                    self.now_data_q = np.load (os.path.join (self.opt.actions_root, file, 'q.npy'))
                    pos_traj, orn_traj = [], []
                    for i in range (self.now_data_q.shape[0]):
                        poses = self.now_data_q[i]
                        for j in range (7):
                            p.resetJointState (self.robotId, j, poses[j], self.now_data_q[i][j])
                        state = p.getLinkState (self.robotId, 7)
                        pos = state[0]
                        orn = state[1]
                        pos_traj.append (pos)
                        orn_traj.append (orn)
                    trajectories.append (np.array (pos_traj))
            dmp_imitation_data = np.array (trajectories)
            self.dmp.imitate (dmp_imitation_data)

    def reset(self):
        try:
            self.destroy()
        except:
            print("initial start!")
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
        # self.evaluator.update (img_path=self.log_path, start_id=0)
        self.seq_num = 0
        self.init_dmp()
        self.init_motion ()
        self.init_rl ()
        self.reset_obj ()

        for j in range (6):
            index_ = self.robot.activeGripperJointIndexList[j]
            p.resetJointState (self.robotId, index_, 0, 0)

        if self.opt.use_embedding:
            action_p = random.random ()

            action_p = int(action_p*len(self.opt.embedding_list))

            if self.opt.nlp_embedding:
                self.opt.load_embedding = self.opt.embedding_list[action_p]
                self.action_embedding = self.embedding_data[self.opt.load_embedding]
            else:
                self.action_embedding = np.array([0]*self.opt.embedding_dim)
                self.action_embedding[action_p] = 1
                self.opt.load_embedding = self.opt.embedding_list[action_p]

        self.init_grasp ()

        img_info = p.getCameraImage (width=self.w,
                                     height=self.h,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     shadow=-1,
                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                     renderer=p.ER_TINY_RENDERER)
        observation = img_info[2][:, :, :3]

        if self.opt.add_mask:
            mask = (img_info[4] > 10000000)
            mask_id_label = [234881025, 301989889, 285212673, 268435457, 318767105, 335544321, 201326593, 218103809,
                             167772161]
            for item in mask_id_label:
                mask = mask * (img_info[4] != item)

            box_mask_label = [117440516,234881028,184549380,33554436,167772164,50331652,
                              134217732,16777220,67108868,150994948,251658244,
                              234881028,100663300,218103812,83886084,268435460]
            for item in box_mask_label:
                mask = mask * (img_info[4] != item)

            observation = cv2.cvtColor (observation, cv2.COLOR_RGB2BGR)
            observation[mask] = [127, 151, 182]

        if self.opt.observation == 'joint_pose':
            observation = np.array([p.getJointState(self.robotId,i)[0] for i in range(self.robotEndEffectorIndex)])
        elif self.opt.observation == 'end_pos':
            observation = np.array(p.getLinkState (self.robotId, 7)[0])
        elif self.opt.observation == 'before_cnn':
            observation = np.array(observation)


        if self.opt.use_embedding:
            self.observation = [self.action_embedding, observation]
        else:
            self.observation = observation

        return self.observation

    def step(self,action):
        self.info = ''
        if self.opt.use_dmp:
            return_value = self.step_dmp(action)
        else:
            return_value = self.step_without_dmp(action)
        return return_value

    def step_dmp(self,action):
        action = action.squeeze()

        init_pos = self.start_pos

        if self.opt.use_embedding:
            self.info += 'target:{}\n'.format(str(self.action_embedding))

        # p.changeVisualShape (self.obj_id, -1, rgbaColor=[0, 0, 1, 1])
        # action[-1] = -0.1
        # orn_traj = np.load (os.path.join (self.env_root, 'init', 'orn.npy'))
        # start_id = 0
        # pos = p.getLinkState (self.robotId, 7)[0]
        # up_traj = point2traj ([pos, [pos[0], pos[1], pos[2] - 0.15]])
        # start_id = self.move (up_traj, orn_traj, start_id)

        self.info += 'action:{}\n'.format(str(action))
        self.dmp.set_start(list(self.start_pos))
        dmp_end_pos = [x+y for x,y in zip(self.start_pos,action)]
        self.dmp.set_goal(dmp_end_pos)
        self.traj = self.dmp.get_traj()

        loose_num = -1
        start_thresh = 0
        if self.opt.add_gripper and action[-1]>start_thresh:
            loose_num = 10
            # if self.opt.load_embedding >= 16 and self.opt.load_embedding <= 20:
            #     loose_num = 5

        dmp_observations = []
        for step_id,small_action in enumerate(self.traj):
            # if out of range, then stop the motion
            for axis_dim in range (3):
                if self.start_pos[axis_dim] < self.axis_limit[axis_dim][0] or \
                        self.start_pos[axis_dim] > self.axis_limit[axis_dim][1]:
                    small_action = np.array([0,0,0])
            # if not add motion as the decision part
            if not self.opt.add_motion:
                small_action = np.array ([0, 0, 0])
            # execute one small step
            small_observation = self.step_without_dmp (small_action)
            dmp_observations.append(small_observation)

            if self.opt.add_gripper and step_id==loose_num:
                gripperPos = 50
                self.robot.gripperControl (gripperPos)

        if self.opt.use_embedding:
            self.observation = [self.action_embedding, dmp_observations[0][0]]
        else:
            self.observation = dmp_observations[0][0]


        reward = dmp_observations[-1][1]
        self.info += 'total reward: {}\n\n'.format (reward)

        done = True
        print(self.info)
        self.log_info.write (self.info)
        return self.observation,reward,done,self.info


    def step_without_dmp(self,action):
        action = action.squeeze()
        self.seq_num += 1
        self.info += 'seq_num:{}\n'.format(self.seq_num)
        self.info += 'now_pos:{}\n'.format(self.start_pos)
        self.info += 'action:{}\n'.format(action)
        self.last_distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5
        pos = [x+y for x,y in zip(self.start_pos,action)]
        orn = self.fix_orn[0]

        # make sure that the error between target pot and real pos is limited within 1 mm
        execute_stage_time = 20
        for execute_t in range(execute_stage_time):
            self.robot.operationSpacePositionControl(pos,orn,self.data_q[0])

        return self.get_observation()

    def get_observation(self):
        # get observation
        img_info = p.getCameraImage (width=self.w,
                                     height=self.h,
                                     viewMatrix=self.view_matrix,
                                     projectionMatrix=self.proj_matrix,
                                     shadow=-1,
                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                     renderer=p.ER_TINY_RENDERER)
        img = img_info[2][:, :, :3]

        if self.opt.add_mask:
            mask = (img_info[4] > 10000000)
            mask_id_label = [234881025, 301989889, 285212673, 268435457, 318767105, 335544321, 201326593, 218103809,
                             167772161]
            for item in mask_id_label:
                mask = mask * (img_info[4] != item)

            box_mask_label = [117440516,234881028,184549380,33554436,167772164,50331652,
                              134217732,16777220,67108868,150994948,251658244,
                              234881028,100663300,218103812,83886084,268435460]
            for item in box_mask_label:
                mask = mask * (img_info[4] != item)

            img = cv2.cvtColor (img, cv2.COLOR_RGB2BGR)
            img[mask] = [127, 151, 182]


        cv2.imwrite (os.path.join (self.log_path, '{:06d}.jpg'.format (self.seq_num - 1)), img)
        self.observation = img

        if self.opt.observation == 'joint_pose':
            self.observation = np.array([p.getJointState (self.robotId, i)[0] for i in range (self.robotEndEffectorIndex)])
        elif self.opt.observation == 'end_pos':
            self.observation = np.array(p.getLinkState (self.robotId, 7)[0])
        elif self.opt.observation == 'before_cnn':
            self.observation = np.array(self.observation)
        self.start_pos =  p.getLinkState(self.robotId,7)[0]

        return self.get_reward()

    def get_reward (self):
        if self.opt.video_reward:
            return self.get_video_reward()
        else:
            return self.get_handcraft_reward()

    def get_video_reward(self):
        if ((self.seq_num-1)%self.opt.give_reward_num==self.opt.give_reward_num-1) \
                and self.seq_num>=self.opt.cut_frame_num:
            if self.opt.use_cycle:
                self.cycle.image_transfer(self.epoch_num)
            self.eval.update(img_path=self.log_path,start_id=self.seq_num-1-self.opt.cut_frame_num)
            self.eval.get_caption()
            rank,probability = self.eval.eval()
            reward = probability
            self.info += 'rank: {}\n'.format(rank)
        else:
            reward = 0

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

        self.info += 'reward: {}\n\n'.format (reward)
        # self.log_info.write (self.info)
        # print (self.info)
        return self.observation, reward, done, self.info

    def get_handcraft_reward(self):
        distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5
        # reward = (0.15 - distance) * 7
        reward = (self.last_distance - distance)*100
        # calculate whether it is done
        if self.seq_num>=self.max_seq_num:
            done = True
        else:
            done = False

        for axis_dim in range(3):
            if self.start_pos[axis_dim]<self.axis_limit[axis_dim][0] or \
                    self.start_pos[axis_dim]>self.axis_limit[axis_dim][1]:
                done = True
                reward = -1

        self.info += 'reward: {}\n\n'.format(reward)
        # self.log_info.write(self.info)
        return self.observation,reward,done,self.info
