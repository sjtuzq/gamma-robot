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
from Solver.TD3_embedding_nlp import TD3_embedding_nlp
from Solver.TD3_new import TD3_new
from Solver.TD3_final import TD3_final
import Envs.bullet_client as bc


from config import opt,device


def main ():
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
    min_Val = torch.tensor (1e-7).float ().to (device)  # min value

    if opt.use_embedding:
        if opt.nlp_embedding:
            # agent = TD3_embedding_nlp(state_dim, action_dim, max_action, env.log_root, opt)
            # agent = TD3_new(state_dim, action_dim, max_action, env.log_root, opt)
            agent = TD3_final(state_dim, action_dim, max_action, env.log_root, opt)
        else:
            agent = TD3_embedding (state_dim, action_dim, max_action, env.log_root, opt)
    else:
        agent = TD3 (state_dim, action_dim, max_action, env.log_root, opt)
    ep_r = 0

    if opt.mode == 'test':
        agent.load (2000)
        for i in range (opt.iteration):
            state = env.reset ()
            for t in range (100):

                action = agent.select_action (state)
                next_state, reward, done, info = env.step (np.float32 (action))
                # if opt.use_embedding:
                #     ep_r += reward[np.where (next_state[0] == 1)[0][0]]
                # else:
                #     ep_r += reward
                # env.render ()
                if done or t == 2000:
                    print ("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format (i, ep_r, t))
                    break
                state = next_state

    elif opt.mode == 'train':
        print ("====================================")
        print ("Collection Experience...")
        print ("====================================")
        # if opt.load: agent.load()

        buffer_root = '/scr1/system/gamma-robot/scripts/utils/buffer/211'
        action_all = np.load(os.path.join(buffer_root,'action_all.npy'))
        target_all = np.load(os.path.join(buffer_root,'target_all.npy'))
        rank_all = np.load(os.path.join(buffer_root,'rank_all.npy'))
        reward_all = np.load(os.path.join(buffer_root,'reward_all.npy'))

        for i in range (opt.num_iteration):
            if i<4000:
                continue
            target = target_all[i]
            state = env.reset (target)

            # file = os.path.join(opt.project_root,'logs','tds_log','test{}'.format(211),'epoch-{}'.format(i))
            # log_reader = open(file,'r')
            # for line in log_reader.readlines():
            #     line = line.strip().split(':')
            #     if line[0]=='target':
            #         target = int(line[1])
            #     if line[0]=='action':

            action = action_all[i]

            print('epoch id:{}, action:{}'.format(i,str(action)))
            next_state, reward, done, info = env.step (action)

            print(reward,reward_all[i],rank_all[i])

    else:
        raise NameError ("mode wrong!!!")

if __name__ == '__main__':
    main()

