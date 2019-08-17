import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import math, os
from config import opt, device
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.cuda_id)


import cv2
import torchvision.transforms as transforms
import pybullet
import torchvision.models as models
import importlib

sys.path.append ('./Eval')
sys.path.append ('./Envs')
sys.path.append ('./Dmp')
sys.path.append ('./Cycle')

from config_a3c import opt, device

from Solver.A3C import A3C_solver
from Solver.A3C_embedding import A3C_solver_embedding
from Solver.A3C_embedding_nlp import A3C_solver_embedding_nlp
from Dmp.gamma_dmp import DMP
from Eval.gamma_pred import Frame_eval
from Cycle.gamma_transfer import Frame_transfer


def main():
    if opt.use_cycle:
        opt.load_cycle = Frame_transfer (opt)

    if opt.use_dmp:
        opt.load_dmp = DMP (opt)
        opt.each_action_lim = opt.each_action_lim * opt.cut_frame_num * opt.dmp_ratio

    if opt.video_reward:
        test_path = os.path.join (opt.project_root, 'logs/a3c_log/test{}'.format (opt.test_id))
        if not os.path.exists (test_path):
            os.mkdir (test_path)
        evaluator = Frame_eval (
            img_path=os.path.join (opt.project_root, 'logs/a3c_log/test{}'.format (opt.test_id), 'epoch-0'),
            frame_len=opt.cut_frame_num,
            start_id=0,
            memory_path=os.path.join (opt.project_root, 'logs/a3c_log/test{}'.format (opt.test_id), 'memory'),
            class_label=opt.action_id,
            opt=opt)
        opt.load_video_pred = evaluator

    Engine_module = importlib.import_module ('Envs.env_{}'.format (opt.action_id))
    RobotEnv = getattr (Engine_module, 'Engine{}'.format (opt.action_id))

    if opt.use_embedding:
        if opt.nlp_embedding:
            agent = A3C_solver_embedding_nlp (opt, RobotEnv)
        else:
            agent = A3C_solver_embedding (opt, RobotEnv)
    else:
        agent = A3C_solver (opt, RobotEnv)

    agent.run ()


if __name__ == "__main__":
    main ()
