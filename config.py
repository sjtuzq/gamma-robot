"""
the whole project only has one absolute path, which is the project_root parameter in opt
quthor: Qiang Zhang
time: 7-28-2019
"""

import argparse
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()

# frequently used parameters
parser.add_argument('--capacity', default=200000, type=int) # replay buffer size
parser.add_argument('--start_train', default=15000, type=int) # replay buffer size
parser.add_argument('--add_buffer', default=True, type=int) # replay buffer size
parser.add_argument('--learning_rate', default=0.000003, type=float)
parser.add_argument('--noise_level', default=0.5, type=float)
parser.add_argument('--noise_training_level', default=0.15, type=float)
parser.add_argument('--batch_size', default=36, type=int) # mini batch size
parser.add_argument('--test_id',  default=222, type=int) #  1000+ means debug
parser.add_argument('--project_root',  default='/scr1/system/gamma-robot/', type=str) # project root path
# parser.add_argument('--project_root',  default='/juno/u/qiangzhang/system/gamma-robot/', type=str) # project root path



# reinforcement learning part hyper parameters
parser.add_argument('--mode', default='test', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v0")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--update_time', default=10, type=int)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor

parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games

parser.add_argument('--seed', default=1, type=int)
# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=200, type=int) #
parser.add_argument('--load', default=True, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.001, type=float)
parser.add_argument('--noise_clip', default=0.03, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.001, type=float)
parser.add_argument('--max_episode', default=2003, type=int)
parser.add_argument('--print_log', default=5, type=int)

parser.add_argument('--align_sample', default=False, type=int)
parser.add_argument('--more_embedding', default=True, type=int)





# environment part hyper parameters
parser.add_argument('--gui',  default=True, type=int) #
parser.add_argument('--video_id',  default=6, type=int) #

parser.add_argument('--object_id',  default='nut', type=str) #
parser.add_argument('--observation',  default='before_cnn', type=str) #joint_pose or end_pos or before_cnn or others
parser.add_argument('--view_point',  default='first', type=str) # first or third
parser.add_argument('--rand_start',  default='fixed', type=str) # rand or two or others
parser.add_argument('--rand_box',  default='fixed', type=str) # rand or two or others

parser.add_argument('--axis_limit_x',  default='[0.04,0.6]', type=str) #
parser.add_argument('--axis_limit_y',  default='[-0.40,0.25]', type=str) #
parser.add_argument('--axis_limit_z',  default='[0.26,0.7]', type=str) #
parser.add_argument('--img_w',  default=320, type=int) #
parser.add_argument('--img_h',  default=240, type=int) #

parser.add_argument('--obj_away_loss',  default=True, type=int) #
parser.add_argument('--away_reward',  default=0, type=float) #
parser.add_argument('--reward_diff',  default=True, type=int) #
parser.add_argument('--out_reward',  default=-10, type=float) #

parser.add_argument('--end_distance',  default=0.20, type=float) #
parser.add_argument('--each_action_lim',  default=0.03, type=float) #
parser.add_argument('--add_gripper',  default=True, type=int) #
parser.add_argument('--add_motion',  default=True, type=int) #
parser.add_argument('--write_img',  default=1, type=int) #
parser.add_argument('--start_write',  default=20000, type=int) #


# video prediction part hyper parameters
parser.add_argument('--action_id',  default=8888, type=int) #
parser.add_argument('--cut_frame_num',  default=20, type=int) #
parser.add_argument('--give_reward_num',  default=1, type=int) #
parser.add_argument('--video_reward',  default=True, type=int) #
parser.add_argument('--load_video_pred',  default=None, type=object) #
parser.add_argument('--add_mask',  default=True, type=int) #
parser.add_argument('--prob_softmax',  default=False, type=int) #
parser.add_argument('--merge_class',  default=True, type=int) #
parser.add_argument('--use_trn',  default=False, type=int) #
parser.add_argument('--use_refine_baseline',  default=False, type=int) #
parser.add_argument('--mean_norm',  default=True, type=int) #
parser.add_argument('--fine_tune',  default=True, type=int) #
parser.add_argument('--fine_tune_list',  default=[86,87,93,94], type=int) #
# parser.add_argument('--fine_tune_list',  default=[8,9,10,11,12], type=int) #
# parser.add_argument('--fine_tune_list',  default=[12], type=int) #


# environment action using DMP part hyperparameters
parser.add_argument('--use_dmp',  default=True, type=int) #
parser.add_argument('--load_dmp', default=None, type=object) #
parser.add_argument('--dmp_ratio', default=0.5, type=float) #
parser.add_argument('--dmp_num', default=40, type=float) #
parser.add_argument('--dmp_imitation', default=False, type=int) #
parser.add_argument('--actions_root', default='/scr1/system/beta-robot/dataset/actions', type=str) #



# environment action using embedding module hyperparameters
parser.add_argument('--use_embedding',  default=True, type=int) #
parser.add_argument('--nlp_embedding',  default=True, type=int) #
# parser.add_argument('--embedding_list',  default=[8,9,10,11,12], type=int) #
# parser.add_argument('--embedding_list',  default=[12,16,86,94,43,45], type=int) #
parser.add_argument('--embedding_list',  default=[86,94,43,45], type=int) #
parser.add_argument('--load_embedding', default=None, type=object) #
parser.add_argument('--embedding_dim', default=4, type=int) #
parser.add_argument('--rl_embedding_dim', default=32, type=int) #
# parser.add_argument('--embedding_adjust', default=True, type=int) #


# environment action using cycle module hyperparameters
parser.add_argument('--use_cycle',  default=False, type=int) #
parser.add_argument('--load_cycle', default=None, type=object) #


# environment action using A3C hyperparameters
parser.add_argument('--use_a3c',  default=False, type=int) #
parser.add_argument('--process_N',  default=1, type=int) #
parser.add_argument('--cuda_id',  default=1, type=int) #


opt = parser.parse_args()

all_task_list = [8,9,10,11,12]
all_task_list.extend([16,17,18,19,20])
all_task_list.extend([40,41,42,43,44,45])
all_task_list.extend([85,86,87,93,94])
all_task_list.extend([100,104,105])


# all_task_list = [85,100,41,44]
# all_task_list = [41,44]

# all_task_list = [40,42,104,105]
# all_task_list = [40,42]

# all_task_list = [86,94,104,105]
# all_task_list = [86,94]


opt.embedding_list = all_task_list

if opt.nlp_embedding:
    opt.embedding_dim = 1024
else:
    opt.embedding_dim = len(opt.embedding_list)



