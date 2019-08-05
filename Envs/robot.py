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

class Robot:
    def __init__(self,pybullet_api,opt,start_pos=[0.4,0.3,0.4]):
       self.p = pybullet_api
       self.opt = opt

       self.gripperMaxForce = 1000.0
       self.armMaxForce = 200.0
       self.endEffectorIndex = 7
       self.start_pos = start_pos

       #lower limits for null space
       self.ll=[-2.9671, -1.8326 ,-2.9671, -3.1416, -2.9671, -0.0873, -2.9671, -0.0001, -0.0001, -0.0001, 0.0, 0.0, -3.14, -3.14, 0.0, 0.0, 0.0, 0.0, -0.0001, -0.0001]
       #upper limits for null space
       self.ul=[2.9671, 1.8326 ,-2.9671, 0.0, 2.9671, 3.8223, 2.9671, 0.0001, 0.0001, 0.0001, 0.81, 0.81, 3.14, 3.14, 0.8757, 0.8757, -0.8, -0.8, 0.0001, 0.0001]
       #joint ranges for null space
       self.jr=[(u-l) for (u,l) in zip(self.ul,self.ll)]

       # restposes for null space
       self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
       # joint damping coefficents
       self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

       self.num_controlled_joints = 7
       self.controlled_joints = [0, 1, 2, 3, 4, 5, 6]

       self.activeGripperJointIndexList = [10, 12, 14, 16, 18, 19]

       self.gripper_left_tip_index = 13
       self.gripper_right_tip_index = 17

       model_path = os.path.join(self.opt.project_root,'scripts','Envs',"urdf/robots/panda/panda_robotiq.urdf")
    
       self.robotId = self.p.loadURDF(model_path, [0, 0, 0]) 
       self.p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0], [0, 0, 0, 1])

       self.targetVelocities = [0] * self.num_controlled_joints
       self.positionGains = [0.03] * self.num_controlled_joints
       self.velocityGains = [1] * self.num_controlled_joints

       self.numJoint = self.p.getNumJoints(self.robotId)

       self.gripperLowerLimitList = []
       self.gripperUpperLimitList = []
       for jointIndex in range(self.numJoint):
            jointInfo = self.p.getJointInfo(self.robotId,jointIndex)
            #print(self.p.getJointInfo(self.robotUid,jointIndex))
            if jointIndex in self.activeGripperJointIndexList:
                self.gripperLowerLimitList.append(jointInfo[8])
                self.gripperUpperLimitList.append(jointInfo[9])
 
    def reset(self): 

        ####### Set Dynamic Parameters for the gripper pad######
        friction_ceof = 1000.0
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, lateralFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, rollingFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, spinningFriction=friction_ceof)

        self.p.changeDynamics(self.robotId, self.gripper_right_tip_index, lateralFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_right_tip_index, rollingFriction=friction_ceof)
        self.p.changeDynamics(self.robotId, self.gripper_left_tip_index, spinningFriction=friction_ceof)

    def jointPositionControl(self,q_list,gripper=None,maxVelocity=None):
        q_list = q_list.tolist()
        if gripper is None:
            self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.controlled_joints,controlMode=self.p.POSITION_CONTROL,targetPositions=q_list)
        else:
            self.gripperOpen = 1 - gripper/255.0
            self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
            self.gripperPos = self.gripperPos.tolist()
            armForce = [self.armMaxForce] * len(self.controlled_joints)
            gripperForce = [self.gripper_max_force] * len(self.activeGripperJointIndexList)
            self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.controlled_joints,controlMode=self.p.POSITION_CONTROL,targetPositions=q_list,forces=armForce)
            self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.activeGripperJointIndexList,controlMode=self.p.POSITION_CONTROL,targetPositions=self.gripperPos,forces=gripperForce)
        self.p.stepSimulation()

    def setJointValue(self,q,gripper):
        for j in range(len(self.controlled_joints)):
            self.p.resetJointState(self.robotId,j,q[j],0.0)
        self.gripperOpen = 1 - gripper/255.0
        self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
        for j in range(6):
            index_ = self.activeGripperJointIndexList[j]
            self.p.resetJointState(self.robotId,index_,self.gripperPos[j],0.0)

    def getEndEffectorPos(self):
        return self.p.getLinkState(self.robotId, self.endEffectorIndex)[0]

    def getEndEffectorVel(self):
        return self.p.getLinkState(self.robotId, self.endEffectorIndex)[6]

    def getGripperTipPos(self):
        left_tip_pos = self.p.getLinkState(self.robotId, self.gripper_left_tip_index)[0]
        right_tip_pos = self.p.getLinkState(self.robotId, self.gripper_right_tip_index)[0]
        gripper_tip_pos = 0.5 * np.array(left_tip_pos) + 0.5 * np.array(right_tip_pos)
        return gripper_tip_pos
   
    def operationSpacePositionControl(self,pos,orn,null_pose=None):
        jointPoses = self.p.calculateInverseKinematics(self.robotId, self.endEffectorIndex, pos, orn,
                                                      lowerLimits=self.ll,
                                                      upperLimits=self.ul,
                                                      jointRanges=self.jr,
                                                      restPoses=null_pose,
                                                      jointDamping=self.jd)[:self.num_controlled_joints]

        self.p.setJointMotorControlArray(bodyIndex=self.robotId,
                                        jointIndices=self.controlled_joints,
                                        controlMode=self.p.POSITION_CONTROL,
                                        targetPositions=jointPoses,
                                        targetVelocities=self.targetVelocities,
                                        forces=[self.armMaxForce] * self.num_controlled_joints)
        self.p.stepSimulation()
     
    def gripperControl(self,gripperPos):
        self.gripperOpen = 1.0 - gripperPos/255.0 
        self.gripperPos = np.array(self.gripperUpperLimitList) * (1 - self.gripperOpen) + np.array(self.gripperLowerLimitList) * self.gripperOpen
        self.gripperPos = self.gripperPos.tolist()
        gripperForce = [self.gripperMaxForce] * len(self.activeGripperJointIndexList)
        self.p.setJointMotorControlArray(bodyUniqueId=self.robotId,jointIndices=self.activeGripperJointIndexList,controlMode=self.p.POSITION_CONTROL,targetPositions=self.gripperPos,forces=gripperForce)
        self.p.stepSimulation()
