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

class Engine:
    def __init__(self,opt):
        self.opt = opt
        self.class_id = opt.action_id
        self.video_id = opt.video_id
        self.test_id = opt.test_id
        self.cut_frame_num = opt.cut_frame_num
        self.give_reward_num = opt.give_reward_num
        self.init_ddpg ()

        if self.opt.video_reward:
            self.eval = self.opt.load_video_pred

        if self.opt.use_cycle:
            self.cycle = self.opt.load_cycle

        if self.opt.use_dmp:
            self.dmp = self.opt.load_dmp
            assert (self.opt.video_reward)

        self.dataset_root = os.path.join(opt.project_root,'dataset')
        # self.dataset_root = self.opt.actions_root.replace('/actions','')
        self.log_root = os.path.join(opt.project_root,'logs')
        self.log_root = safe_path(self.log_root+'/td3_log/test{}'.format(self.test_id))

        self.env_root = os.path.join(opt.project_root,'scripts','Envs')
        self.script_root = os.path.join(opt.project_root,'scripts')
        self.memory_path = safe_path(os.path.join(self.log_root,'memory'))
        backup_code (self.script_root, self.log_root)

    def destroy(self):
        p.disconnect(self.physical_id)

    def init_table(self):
        table_path = os.path.join(self.env_root,'urdf/table/table.urdf')
        self.table_id = p.loadURDF(table_path, [0.42,0,0],[0,0,math.pi*0.32,1],globalScaling=0.6,physicsClientId=self.physical_id)
        texture_path = os.path.join(self.env_root,'texture/table_textures/table_texture.jpg')
        table_textid = p.loadTexture (texture_path)
        p.changeVisualShape (self.table_id, -1, textureUniqueId=table_textid)

    def init_motion(self):
        self.motion_path = self.dataset_root+'/actions/{}-{}'.format(self.class_id,self.video_id)
        self.data_q = np.load (self.dataset_root+'/actions/{}-{}/q.npy'.format(self.class_id,self.video_id))
        self.data_dq = np.load(self.dataset_root+'/actions/{}-{}/dq.npy'.format(self.class_id,self.video_id))
        self.data_gripper = np.load(self.dataset_root+'/actions/{}-{}/gripper.npy'.format(self.class_id,self.video_id))
        self.frames_path = cut_frame(self.dataset_root+'/actions/{}-{}/{}-{}.avi'.format (self.class_id, self.video_id,self.class_id,self.video_id),
                                     self.dataset_root+'/actions/{}-{}/frames'.format (self.class_id, self.video_id))
        self.output_path = safe_path(self.dataset_root+'/actions/{}-{}/simulation_frames'.format(self.class_id,self.video_id))
        self.mask_frames_path = safe_path(self.dataset_root+'/actions/{}-{}/masked_frames'.format(self.class_id,self.video_id))

    def init_plane(self):
        self.plane_id = p.loadURDF ("plane.urdf", [0.7, 0, 0], [0, 0, -math.pi * 0.02, 1], globalScaling=0.7)
        texture_path = os.path.join(self.opt.project_root,'scripts','Envs','texture/real_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        textid = p.loadTexture(texture_file)
        p.changeVisualShape (self.plane_id, -1, rgbaColor=[1, 1, 1, 0.9])
        p.changeVisualShape (self.plane_id, -1, textureUniqueId=textid)

    def init_robot(self):
        # load robot arm and gripper
        model_path = os.path.join(self.opt.project_root,'scripts','Envs',"urdf/robots/panda/panda_robotiq.urdf")
        self.kukaId = p.loadURDF(model_path, [0, 0, -0.2])
        p.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0], [0, 0, 0, 1])
        self.numJoints = p.getNumJoints(self.kukaId)
        self.kukaEndEffectorIndex = 7


        # set up some parameters
        # ik_solver
        self.ik_solver = p.IK_DLS
        # lower limits for null space
        self.ll = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, -0.0001, -0.0001, -0.0001, 0.0, 0.0,
                   -3.14, -3.14, 0.0, 0.0, 0.0, 0.0, -0.0001, -0.0001]
        # upper limits for null space
        self.ul = [2.9671, 1.8326, -2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0001, 0.0001, 0.0001, 0.81, 0.81, 3.14, 3.14,
                   0.8757, 0.8757, -0.8, -0.8, 0.0001, 0.0001]
        # joint ranges for null space
        self.jr = [(u - l) for (u, l) in zip (self.ul, self.ll)]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
        self.num_controlled_joints = self.kukaEndEffectorIndex - 1
        self.controlled_joints = list(range(self.kukaEndEffectorIndex - 1))
        self.targetVelocities = [0] * self.num_controlled_joints
        self.forces = [500] * self.num_controlled_joints
        # self.forces = [0.0] * self.num_controlled_joints
        self.positionGains = [0.03] * self.num_controlled_joints
        self.velocityGains = [1] * self.num_controlled_joints

        self.view_matrix, self.proj_matrix = get_view (self.opt)
        self.q_home = np.array((0., -np.pi/6., 0., -5./6.*np.pi, 0., 2./3.*np.pi, 0.))
        self.idx_ee = self.numJoints - 6
        self.w = self.opt.img_w
        self.h = self.opt.img_h

    def init_gripper(self):
        self.activeGripperJointIndexList = [10, 12, 14, 16, 18, 19]
        self.numJoint = p.getNumJoints (self.kukaId)
        self.gripperLowerLimitList = []
        self.gripperUpperLimitList = []
        for jointIndex in range (self.numJoint):
            jointInfo = p.getJointInfo (self.kukaId, jointIndex)
            if jointIndex in self.activeGripperJointIndexList:
                self.gripperLowerLimitList.append (jointInfo[8])
                self.gripperUpperLimitList.append (jointInfo[9])

        self.gripper_left_tip_index = 13  # 14
        self.gripper_right_tip_index = 17  # 16
        self.gripperForce = 1000

        friction_ceof = 1000.0
        p.changeDynamics (self.kukaId, self.gripper_left_tip_index, lateralFriction=friction_ceof)
        p.changeDynamics (self.kukaId, self.gripper_left_tip_index, rollingFriction=friction_ceof)
        p.changeDynamics (self.kukaId, self.gripper_left_tip_index, spinningFriction=friction_ceof)

        p.changeDynamics (self.kukaId, self.gripper_right_tip_index, lateralFriction=friction_ceof)
        p.changeDynamics (self.kukaId, self.gripper_right_tip_index, rollingFriction=friction_ceof)
        p.changeDynamics (self.kukaId, self.gripper_left_tip_index, spinningFriction=friction_ceof)

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
            self.obj_id = p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling,physicsClientId=self.physical_id)

        if self.opt.object_id == 'cup':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/urdfs/cup.urdf")
            self.obj_position = [0.45, -0.18, 0.34]
            # self.obj_position = [0.55, 0, 0.34]
            # self.obj_position = [0.40, -0.15, 0.34]
            self.obj_orientation = p.getQuaternionFromEuler([-math.pi/2, 0, 0])
            # self.obj_orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
            self.obj_scaling = 0.11
            self.obj_id = p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling,physicsClientId=self.physical_id)

        if self.opt.object_id == 'nut':
            self.obj_file = os.path.join(self.env_root,"urdf/objmodels/nut.urdf")
            # self.obj_position = [0.45, -0.15, 0.4]
            self.obj_position = [0.4, -0.15, 0.34]
            # self.obj_orientation = p.getQuaternionFromEuler([-math.pi/2, 0, 0])
            self.obj_scaling = 2
            self.obj_orientation = p.getQuaternionFromEuler([math.pi/2, -math.pi/2, 0])
            self.obj_id = p.loadURDF(fileName=self.obj_file, basePosition=self.obj_position,baseOrientation=self.obj_orientation,
                                     globalScaling=self.obj_scaling,physicsClientId=self.physical_id)

        texture_path = os.path.join(self.env_root,'texture/sun_textures')
        texture_file = os.path.join(texture_path,random.sample(os.listdir(texture_path),1)[0])
        textid = p.loadTexture(texture_file)
        # p.changeVisualShape (self.plane_id, -1, rgbaColor=[1, 1, 1, 0.9])
        p.changeVisualShape (self.obj_id, -1, textureUniqueId=textid)

    def run(self):
        for i in range(self.data_q.shape[0]):
            jointPoses = self.data_q[i]
            for j in range(self.kukaEndEffectorIndex):
                p.resetJointState(self.kukaId, j, jointPoses[j], self.data_dq[i][j])

            gripper = self.data_gripper[i]
            self.gripperOpen = 1 - gripper / 255.0
            self.gripperPos = np.array (self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array (
                self.gripperLowerLimitList) * self.gripperOpen
            for j in range (6):
                index_ = self.activeGripperJointIndexList[j]
                p.resetJointState (self.kukaId, index_, self.gripperPos[j], 0)

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
                p.resetJointState (self.kukaId, j, poses[j], self.data_dq[i][j])

            state = p.getLinkState (self.kukaId, 7)
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
            img_info = p.getCameraImage (width=self.w,
                                         height=self.h,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=-1,
                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                         renderer=p.ER_TINY_RENDERER)
            # self.save_video (img_info, start_id)
            start_id += 1

        pos = p.getLinkState (self.kukaId, 7)[0]
        left_traj = point2traj([pos, [pos[0], pos[1]+0.14, pos[2]+0.05]])
        start_id = self.core(left_traj, orn_traj,start_id)

        self.start_pos = p.getLinkState(self.kukaId,7)[0]

    def move_up(self):
        # move in z-axis direction
        orn_traj = np.load (os.path.join(self.env_root,'init','orn.npy'))
        pos = p.getLinkState (self.kukaId, 7)[0]
        up_traj = point2traj ([pos, [pos[0], pos[1], pos[2] + 0.3]],delta=0.005)
        start_id = self.core (up_traj, orn_traj,0)

    def explore(self,traj):
        orn_traj = np.load ('orn.npy')
        start_id = self.core (traj, orn_traj,0)

    def core(self,pos_traj,orn_traj,start_id=0):
        for i in range(int(len(pos_traj))):
            pos = pos_traj[i]
            orn = orn_traj[i]
            jointPoses = p.calculateInverseKinematics(self.kukaId, self.kukaEndEffectorIndex, pos, orn,
                                                      lowerLimits=self.ll,
                                                      upperLimits=self.ul,
                                                      jointRanges=self.jr,
                                                      restPoses=self.data_q[i],
                                                      jointDamping=self.jd)[:self.num_controlled_joints]

            p.setJointMotorControlArray(bodyIndex=self.kukaId,
                                        jointIndices=self.controlled_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=jointPoses,
                                        targetVelocities=self.targetVelocities,
                                        forces=self.forces)

            p.stepSimulation()
            img_info = p.getCameraImage (width=self.w,
                                         height=self.h,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix,
                                         shadow=-1,
                                         flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                         renderer=p.ER_TINY_RENDERER)
            # self.save_video(img_info,start_id+i)
        return start_id+len(pos_traj)

    def init_ddpg(self):
        self.target_pos = [0.3633281737186908, -0.23858468351424078, 0.670415682662147]
        # self.target_pos = [0.3633281737186908, -0.23858468351424078+0.1, 0.670415682662147-0.1]
        if self.opt.observation == 'after_cnn':
            self.observation_space = 32
        elif self.opt.observation == 'joint_pose':
            self.observation_space = 7
        elif self.opt.observation == 'end_pos':
            self.observation_space = 3
        elif self.opt.observation == 'before_cnn':
            self.observation_space = 256
        else:
            self.observation_space = 32
        self.each_action_lim = self.opt.each_action_lim
        low = [-self.each_action_lim]*3
        high = [self.each_action_lim]*3
        self.action_space = {'low':low,'high':high}
        self.max_seq_num = 500
        self.min_dis_lim = self.opt.end_distance
        self.axis_limit = [eval(self.opt.axis_limit_x),eval(self.opt.axis_limit_y),
                           eval(self.opt.axis_limit_z)]
        self.cnn = CNN()

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
                            p.resetJointState (self.kukaId, j, poses[j], self.now_data_q[i][j])
                        state = p.getLinkState (self.kukaId, 7)
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
        if self.opt.gui:
            self.physical_id = p.connect(p.GUI)
        else:
            self.physical_id = p.connect (p.DIRECT)

        self.use_gpu = False
        if self.use_gpu:
            plugin = p.loadPlugin (egl.get_filename (), "_eglRendererPlugin")
            print ("plugin=", plugin)
            p.configureDebugVisualizer (p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer (p.COV_ENABLE_GUI, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep (1/250.)
        try:
            self.epoch_num += 1
        except:
            self.epoch_num = 0

        self.log_path = safe_path(os.path.join(self.log_root,'epoch-{}'.format(self.epoch_num)))
        self.log_info = open(os.path.join(self.log_root,'epoch-{}.txt'.format(self.epoch_num)),'w')
        # self.evaluator.update (img_path=self.log_path, start_id=0)
        self.seq_num = 0
        self.init_robot ()
        self.init_dmp()
        self.init_plane ()
        self.init_motion ()
        self.init_gripper ()
        self.init_table ()
        self.init_ddpg ()
        self.init_obj ()
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
            observation = cv2.cvtColor (observation, cv2.COLOR_RGB2BGR)
            observation[mask] = [127, 151, 182]

        if self.opt.observation == 'after_cnn':
            observation = self.cnn(torch.tensor(observation).unsqueeze(0).transpose(1,3).float()).squeeze().data.numpy()
        elif self.opt.observation == 'joint_pose':
            observation = np.array([p.getJointState(self.kukaId,i)[0] for i in range(self.kukaEndEffectorIndex)])
        elif self.opt.observation == 'end_pos':
            observation = np.array(p.getLinkState (self.kukaId, 7)[0])
        elif self.opt.observation == 'before_cnn':
            observation = np.array(observation)
        return observation

    def step(self,action):
        if self.opt.use_dmp:
            return_value = self.step_dmp(action)
        else:
            return_value = self.step_without_dmp(action)
        return return_value

    def step_dmp(self,action):
        action = action.squeeze()
        self.dmp.set_start(list(self.start_pos))
        dmp_end_pos = [x+y for x,y in zip(self.start_pos,action)]
        self.dmp.set_goal(dmp_end_pos)
        self.traj = self.dmp.get_traj()

        dmp_observations = []
        for small_action in self.traj:
            # if out of range, then stop the motion
            for axis_dim in range (3):
                if self.start_pos[axis_dim] < self.axis_limit[axis_dim][0] or \
                        self.start_pos[axis_dim] > self.axis_limit[axis_dim][1]:
                    small_action = np.array([0,0,0])
            small_observation = self.step_without_dmp (small_action)
            dmp_observations.append(small_observation)

        self.observation = dmp_observations[0][0]
        reward = dmp_observations[-1][1]
        done = True
        return self.observation,reward,done,self.info


    def step_without_dmp(self,action):
        action = action.squeeze()
        self.seq_num += 1
        self.info = ''
        self.info += 'seq_num:{}\n'.format(self.seq_num)
        self.info += 'now_pos:{}\n'.format(self.start_pos)
        self.info += 'action:{}\n'.format(action)
        self.last_distance = sum ([(x - y) ** 2 for x, y in zip (self.start_pos, self.target_pos)]) ** 0.5
        pos = [x+y for x,y in zip(self.start_pos,action)]
        orn = self.fix_orn[0]

        # make sure that the error between target pot and real pos is limited within 1 mm
        execute_stage_time = 20
        for execute_t in range(execute_stage_time):
            jointPoses = p.calculateInverseKinematics (self.kukaId, self.kukaEndEffectorIndex, pos, orn,
                                                       lowerLimits=self.ll,
                                                       upperLimits=self.ul,
                                                       jointRanges=self.jr,
                                                       # restPoses=self.now_data_q[0],
                                                       jointDamping=self.jd)[:self.num_controlled_joints]

            p.setJointMotorControlArray (bodyIndex=self.kukaId,
                                         jointIndices=self.controlled_joints,
                                         controlMode=p.POSITION_CONTROL,
                                         targetPositions=jointPoses,
                                         targetVelocities=self.targetVelocities,
                                         forces=self.forces)
            p.stepSimulation()

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
            img = cv2.cvtColor (img, cv2.COLOR_RGB2BGR)
            img[mask] = [127, 151, 182]

        cv2.imwrite (os.path.join (self.log_path, '{:06d}.jpg'.format (self.seq_num - 1)), img)
        self.observation = img
        if self.opt.observation == 'after_cnn':
            self.observation = self.cnn (torch.tensor (self.observation).unsqueeze (0).transpose (1, 3).float ()).squeeze ().data.numpy ()
        elif self.opt.observation == 'joint_pose':
            self.observation = np.array([p.getJointState (self.kukaId, i)[0] for i in range (self.kukaEndEffectorIndex)])
        elif self.opt.observation == 'end_pos':
            self.observation = np.array(p.getLinkState (self.kukaId, 7)[0])
        elif self.opt.observation == 'before_cnn':
            self.observation = np.array(self.observation)
        self.start_pos =  p.getLinkState(self.kukaId,7)[0]

        return self.get_reward()

    def get_reward(self):
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
        self.log_info.write(self.info)
        print(self.info)
        return self.observation,reward,done,self.info
