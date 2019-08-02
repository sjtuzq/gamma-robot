"""
    debug the module of cycle-gan
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./Eval')
sys.path.append('./Envs')
sys.path.append('./Dmp')
sys.path.append('./Cycle')


from Dmp.gamma_dmp import DMP
from Eval.gamma_pred import Frame_eval
from Cycle.gamma_transfer import Frame_transfer

from Envs.env_107 import Engine107
from Envs.env_106 import Engine106
from Envs.env_45 import Engine45
from Solver.TD3 import TD3

from config import opt,device



def main ():
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

    env = eval('Engine{} (opt)'.format(opt.action_id))

    state_dim = env.observation_space
    action_dim = len (env.action_space['high'])
    max_action = env.action_space['high'][0]
    min_Val = torch.tensor (1e-7).float ().to (device)  # min value

    agent = TD3 (state_dim, action_dim, max_action, env.log_root, opt)
    ep_r = 0

    if opt.mode == 'test':
        agent.load ()
        for i in range (opt.iteration):
            state = env.reset ()
            for t in range (100):
                action = agent.select_action (state)
                next_state, reward, done, info = env.step (np.float32 (action))
                ep_r += reward
                env.render ()
                if done or t == 2000:
                    print ("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format (i, ep_r, t))
                    break
                state = next_state

    elif opt.mode == 'train':
        print ("====================================")
        print ("Collection Experience...")
        print ("====================================")
        # if opt.load: agent.load()
        for i in range (opt.num_iteration):
            state = env.reset ()
            for t in range (2000):

                action = agent.select_action (state)
                action = action + np.random.normal (0, max_action * opt.noise_level, size=action.shape)
                action = action.clip (env.action_space['low'], env.action_space['high'])
                next_state, reward, done, info = env.step (action)

                ep_r += reward
                # if opt.render and i >= opt.render_interval : env.render()
                agent.memory.push ((state, next_state, action, reward, np.float (done)))
                if i + 1 % 10 == 0:
                    print ('Episode {},  The memory size is {} '.format (i, len (agent.memory.storage)))
                if len (agent.memory.storage) >= opt.start_train - 1:
                    agent.update (opt.update_time)

                state = next_state
                if done or t == opt.max_episode - 1:
                    agent.writer.add_scalar ('ep_r', ep_r, global_step=i)
                    if i % opt.print_log == 0:
                        print ("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format (i, ep_r, t))
                    ep_r = 0
                    break

            if i % opt.log_interval == 0:
                agent.save ()

    else:
        raise NameError ("mode wrong!!!")

if __name__ == '__main__':
    main()

