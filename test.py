"""
test envs
test rl algorithms
test the whole project frameworks
"""
import os
import sys
sys.path.append('./Envs')
sys.path.append('./Solver')

import numpy as np
import torch

from config import device,opt
from Envs.env_106 import Engine106 as Engine
from Solver.TD3 import TD3

def test_config():
    print(opt.project_root)

def test_import_by_path():
    import importlib.util
    spec = importlib.util.spec_from_file_location (os.path.join(opt.project_root,'scripts','reference','envs','utils.py'))
    foo = importlib.util.module_from_spec (spec)
    spec.loader.exec_module (foo)
    foo.MyClass ()

    # from importlib import import_module
    # module_path = '/scr1/system/gamma-robot/scripts/reference/envs'
    # # module_path = os.path.join(opt.project_root,'scripts','reference','envs','utils.py')
    # module = import_module(module_path)


def test_env():
    agent = Engine(opt)
    agent.reset()
    action = np.array([0,0.01,0.03])
    agent.step(action)


def main():
    env = Engine (opt)

    state_dim = env.observation_space
    action_dim = len (env.action_space['high'])
    max_action = env.action_space['high'][0]
    min_Val = torch.tensor (1e-7).float ().to (device)  # min value

    agent = TD3(state_dim, action_dim, max_action,env.log_root,opt)
    ep_r = 0

    if opt.mode == 'test':
        agent.load()
        for i in range(opt.iteration):
            state = env.reset()
            for t in range(100):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t ==2000 :
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    break
                state = next_state

    elif opt.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        # if opt.load: agent.load()
        for i in range(opt.num_iteration):
            state = env.reset()
            for t in range(2000):

                action = agent.select_action(state)
                action = action + np.random.normal(0, max_action*opt.noise_level, size=action.shape)
                action = action.clip(env.action_space['low'], env.action_space['high'])
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                # if opt.render and i >= opt.render_interval : env.render()
                agent.memory.push((state, next_state, action, reward, np.float(done)))
                if i+1 % 10 == 0:
                    print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                if len(agent.memory.storage) >= opt.capacity-1:
                    agent.update(opt.update_time)

                state = next_state
                if done or t == opt.max_episode -1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % opt.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % opt.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")
    
if __name__ == '__main__':
    main()