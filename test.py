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
    assert (opt.mode=='test')
    if opt.mode == 'test':
        opt.test_id = 8888

    Engine_module = importlib.import_module ('Envs.env_{}'.format (opt.action_id))
    Engine = getattr (Engine_module, 'Engine{}'.format (opt.action_id))
    if opt.use_cycle:
        opt.load_cycle = Frame_transfer (opt)

    if opt.use_dmp:
        opt.load_dmp = DMP (opt)
        opt.each_action_lim = opt.each_action_lim * opt.cut_frame_num * opt.dmp_ratio

    if opt.video_reward:
        test_path = os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id))
        if not os.path.exists (test_path):
            os.mkdir (test_path)
        evaluator = Frame_eval (
            img_path=os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id), 'epoch-0'),
            frame_len=opt.cut_frame_num,
            start_id=0,
            memory_path=os.path.join (opt.project_root, 'logs/td3_log/test{}'.format (opt.test_id), 'memory'),
            class_label=opt.action_id,
            opt=opt)
        opt.load_video_pred = evaluator

    if opt.gui:
        opt.p = bc.BulletClient (connection_mode=pybullet.GUI)
    else:
        opt.p = bc.BulletClient (connection_mode=pybullet.DIRECT)

    env = eval ('Engine(opt)'.format (opt.action_id))

    state_dim = env.observation_space
    action_dim = len (env.action_space['high'])
    max_action = env.action_space['high'][0]
    min_Val = torch.tensor (1e-7).float ().to (device)  # min value

    if opt.use_embedding:
        if opt.nlp_embedding:
            # agent = TD3_embedding_nlp(state_dim, action_dim, max_action, env.log_root, opt)
            # agent = TD3_new(state_dim, action_dim, max_action, env.log_root, opt)
            agent = TD3_final (state_dim, action_dim, max_action, env.log_root, opt)
        else:
            agent = TD3_embedding (state_dim, action_dim, max_action, env.log_root, opt)
    else:
        agent = TD3 (state_dim, action_dim, max_action, env.log_root, opt)


    if opt.mode == 'test':
        weight_id = 1400
        test_file = open(os.path.join(test_path,'test_{}.txt'.format(weight_id)),'w')
        # agent.load (4400)
        agent.load (weight_id)

        for target in opt.embedding_list:
            state = env.reset (target)
            action = agent.select_action (state)
            action = action.clip (-max_action, max_action)
            next_state, reward, done, info = env.step (action)

            reward_id = np.where (np.array (env.opt.embedding_list) == env.opt.load_embedding)[0][0]
            test_file.write('{}\n'.format(reward[reward_id]))
            print(action)


    else:
        raise NameError ("mode wrong!!!")


if __name__ == '__main__':
    main()

