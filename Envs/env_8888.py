#!/usr/bin/env python3
"""
   env for action 8000, consist of the following different tasks:
   action 8,9,10,11,12,   16,17,18,19,20,  40,41,42,43,44,45   85,86,87,93,94  100,104,105,106,107
"""

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

import pkgutil
egl = pkgutil.get_loader ('eglRenderer')

from .env import Engine
from .merge_grasp import *

class Engine8888(Engine):
    def __init__(self, opt, worker_id=None,p_id=None):
        super(Engine8888,self).__init__(opt=opt,p_id=p_id,w_id=worker_id)
        self.opt = opt
        self._wid = worker_id

    def init_grasp(self):
        p.resetBasePositionAndOrientation (self.table_id, [0.42, 0, 0], [0, 0, math.pi * 0.32, 1])
        try:
            p.removeBody (self.box_id)
        except:
            pass

        now_task_id = self.opt.load_embedding
        cmd = 'init_grasp_{}(self=self)'.format(now_task_id)
        eval(cmd)
        # init_grasp_8(self=self)