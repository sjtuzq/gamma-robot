import os
import sys
import torch
import numpy as np
import pybullet
import matplotlib.pyplot as plt
import importlib

sys.path.append('./Eval')
sys.path.append('./Envs')
sys.path.append('./Dmp')
sys.path.append('./Cycle')


from Dmp.gamma_dmp import DMP
from Eval.gamma_pred import Frame_eval
from Cycle.gamma_transfer import Frame_transfer
from Solver.TD3 import TD3
from Solver.TD3_embedding import TD3_embedding
import Envs.bullet_client as bc


from config import opt,device

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def show_one_policy ():
    Engine_module = importlib.import_module('Envs.env_{}'.format(opt.action_id))
    Engine = getattr(Engine_module,'Engine{}'.format(opt.action_id))
    if opt.use_cycle:
        opt.load_cycle = Frame_transfer (opt)

    if opt.use_dmp:
        opt.load_dmp = DMP(opt)
        opt.each_action_lim = opt.each_action_lim*opt.cut_frame_num*opt.dmp_ratio

    if opt.video_reward:
        test_path = os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id))
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        evaluator = Frame_eval (img_path=os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id), 'epoch-0'),
                               frame_len=opt.cut_frame_num,
                               start_id=0,
                               memory_path=os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id), 'memory'),
                               class_label=opt.action_id,
                               opt = opt)
        opt.load_video_pred = evaluator

    if opt.gui:
        opt.p = bc.BulletClient (connection_mode=pybullet.GUI)
    else:
        opt.p = bc.BulletClient (connection_mode=pybullet.DIRECT)

    env = eval('Engine(opt)'.format(opt.action_id))

    state_dim = env.observation_space
    action_dim = len (env.action_space['high'])
    max_action = env.action_space['high'][0]

    if opt.use_embedding:
        agent = TD3_embedding (state_dim, action_dim, max_action, env.log_root, opt)
    else:
        agent = TD3 (state_dim, action_dim, max_action, env.log_root, opt)


    assert(opt.mode == 'test')
    agent.load (2000)
    state = env.reset ()

    inter_n = 10.

    fig = plt.figure ()
    ax = Axes3D (fig)
    # ax.view_init(elev=45,azim=0)
    X = np.arange (0, 1+1/inter_n, 1/inter_n)
    Y = np.arange (0, 1+1/inter_n, 1/inter_n)
    X, Y = np.meshgrid (X, Y)
    R = np.sqrt (X ** 2 + Y ** 2)
    Z = np.sin (R)

    help(ax.plot_surface)

    for i in range(int(inter_n+1)):
        for j in range(int(inter_n+1)):
            state[0] = np.array ([i/inter_n, 0, 0, j/inter_n])
            action = agent.select_action (state)
            obj = env.p.getAABB (env.obj_id, -1)
            obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
            world_pos = [(x + y) for x, y in zip (obj_center, action)]
            world_pos.append (1)
            camera_pos = np.array (env.view_matrix).reshape (4, -1).T.dot (np.array (world_pos))
            camera_pos = [x / camera_pos[-1] for x in camera_pos]

            print((i,j),camera_pos)
            Z[i][j] = camera_pos[1]

    ax.plot_surface (X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show ()

    plt.cla ()
    fig = plt.figure ()
    ax = Axes3D (fig)
    ax.view_init (elev=45, azim=90)
    ax.plot_surface (X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show ()


def show_rgb ():
    Engine_module = importlib.import_module('Envs.env_{}'.format(opt.action_id))
    Engine = getattr(Engine_module,'Engine{}'.format(opt.action_id))
    if opt.use_cycle:
        opt.load_cycle = Frame_transfer (opt)

    if opt.use_dmp:
        opt.load_dmp = DMP(opt)
        opt.each_action_lim = opt.each_action_lim*opt.cut_frame_num*opt.dmp_ratio

    if opt.video_reward:
        test_path = os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id))
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        evaluator = Frame_eval (img_path=os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id), 'epoch-0'),
                               frame_len=opt.cut_frame_num,
                               start_id=0,
                               memory_path=os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id), 'memory'),
                               class_label=opt.action_id,
                               opt = opt)
        opt.load_video_pred = evaluator

    if opt.gui:
        opt.p = bc.BulletClient (connection_mode=pybullet.GUI)
    else:
        opt.p = bc.BulletClient (connection_mode=pybullet.DIRECT)

    env = eval('Engine(opt)'.format(opt.action_id))

    state_dim = env.observation_space
    action_dim = len (env.action_space['high'])
    max_action = env.action_space['high'][0]

    if opt.use_embedding:
        agent = TD3_embedding (state_dim, action_dim, max_action, env.log_root, opt)
    else:
        agent = TD3 (state_dim, action_dim, max_action, env.log_root, opt)


    assert(opt.mode == 'test')
    agent.load (2000)
    state = env.reset ()

    inter_n = 100

    # fig = plt.figure ()
    # ax = Axes3D (fig)
    # # ax.view_init(elev=45,azim=0)
    # X = np.arange (0, 1+1/inter_n, 1/inter_n)
    # Y = np.arange (0, 1+1/inter_n, 1/inter_n)
    # X, Y = np.meshgrid (X, Y)
    # R = np.sqrt (X ** 2 + Y ** 2)
    # Z = np.sin (R)
    #
    # help(ax.plot_surface)

    img = np.zeros((int(inter_n+1),int(inter_n+1),3))
    world_data = np.zeros((int(inter_n+1),int(inter_n+1),3))
    camera_data = np.zeros((int(inter_n+1),int(inter_n+1),3))

    for i in range(int(inter_n+1)):
        for j in range(int(inter_n+1)):
            state[0] = np.array ([i/inter_n, 0, 0, j/inter_n])
            action = agent.select_action (state)
            obj = env.p.getAABB (env.obj_id, -1)
            obj_center = [(x + y) * 0.5 for x, y in zip (obj[0], obj[1])]
            world_pos = [(x + y) for x, y in zip (obj_center, action)]
            world_pos.append (1)
            camera_pos = np.array (env.view_matrix).reshape (4, -1).T.dot (np.array (world_pos))
            camera_pos = [x / camera_pos[-1] for x in camera_pos]

            # img[i][j] = world_pos[:3]
            # img[i][j] = camera_pos[:3]
            world_data[i][j] = world_pos[:3]
            camera_data[i][j] = camera_pos[:3]

            print(i,j,camera_pos)

    np.save('utils/logs/reward_data_31_world_1.npy',world_data)
    np.save('utils/logs/reward_data_31_camera_1.npy',camera_data)

    # for i in range (3):
    #     img[:, :, i] = (img[:, :, i] - img[:, :, i].min ())
    #     img[:,:,i] = img[:,:,i]* 1 / img[:, :, i].max ()
    # plt.cla ()
    # plt.imshow (img)
    # plt.show ()

if __name__ == '__main__':
    show_rgb()

