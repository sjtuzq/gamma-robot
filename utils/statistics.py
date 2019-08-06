import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_end_pos(test_id=92,trained=True,flag=50):
    project_root = '/scr1/system/gamma-robot/'
    log_dir = os.path.join(project_root,'logs','td3_log','test{}'.format(test_id))

    pos_data = []
    reward_data = []
    rank_data = []
    for file in os.listdir(log_dir):
        if '.txt' in file:
            epoch_id = int(file.split('.')[0].split('-')[1])
            if trained and epoch_id<flag:
                continue
            if (not trained) and epoch_id>flag:
                continue
            with open(os.path.join(log_dir,file)) as f:
                for line in f.readlines():
                    if 'now_pos' in line:
                        pos = eval(line.strip().split(':')[1])
                    if 'reward' in line:
                        reward = float(line.strip().split(':')[1])
                    if 'rank' in line:
                        rank = float(line.strip().split(':')[1])
            pos_data.append(list(pos))
            reward_data.append(reward)
            rank_data.append(rank)

    pos_data = np.array(pos_data)
    reward_data = np.array(reward_data)
    rank_data = np.array(rank_data)
    print(rank_data.mean(),reward_data.mean())

    reward_data = reward_data-reward_data.mean()
    reward_rank = np.argsort(reward_data)
    C = []
    for item in reward_rank:
        if item<10:
            C.append('r')
        else:
            C.append('g')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pos_data[:,0],pos_data[:,1],pos_data[:,2],c=C)

    if not os.path.exists('end_pos_log'):
        os.mkdir('end_pos_log')
    # plt.show()
    if trained:
        plt.savefig('end_pos_log/test_{}_after.jpg'.format(test_id))
    else:
        plt.savefig ('end_pos_log/test_{}_before.jpg'.format(test_id))

def show_dmp_effeciency(test_id=100,flag=100):
    # show_end_pos(test_id=90,trained=False,flag=50)
    # show_end_pos(test_id=90,trained=True,flag=50)
    #
    # show_end_pos(test_id=91,trained=False,flag=100)
    # show_end_pos(test_id=91,trained=True,flag=100)
    #
    # show_end_pos(test_id=92,trained=False,flag=50)
    # show_end_pos(test_id=92,trained=True,flag=50)

    # show_end_pos(test_id=103,trained=False,flag=50)
    # show_end_pos(test_id=103,trained=True,flag=50)

    # show_end_pos (test_id=105, trained=False, flag=50)
    # show_end_pos (test_id=105, trained=True, flag=50)

    show_end_pos (test_id=test_id, trained=False, flag=flag)
    show_end_pos (test_id=test_id, trained=True, flag=flag)



def show_cycle_effeciency(test_id=1000,use_rank=True):
    rank_diff = 6
    prob_diff = 0.8
    project_root = '/scr1/system/gamma-robot/'
    log_dir = os.path.join(project_root,'logs','td3_log','test{}'.format(test_id))

    higher_list = []
    lower_list = []
    rank = 0
    before_rank = 0
    for file in os.listdir(log_dir):
        if '.txt' in file:
            epoch_id = int(file.split('.')[0].split('-')[1])
            with open(os.path.join(log_dir,file)) as f:
                for line in f.readlines():
                    if 'before_rank' in line:
                        before_rank = int(line.strip().split(' ')[0].split(':')[1])
                        before_probability = float(line.strip().split(' ')[-1].split(':')[1])
                    if 'rank' in line:
                        rank = int (line.strip ().split (' ')[0].split (':')[1])
                        probability = float(line.strip ().split (' ')[-1].split (':')[1])
                if use_rank:
                    if rank-before_rank>rank_diff:
                        lower_list.append(epoch_id)
                    if before_rank-rank>rank_diff:
                        higher_list.append(epoch_id)
                else:
                    if probability-before_probability>prob_diff:
                        higher_list.append(epoch_id)
                    if before_probability-probability>prob_diff:
                        lower_list.append(epoch_id)

    print('higher_list:{}',str(higher_list))
    print('lower_list:{}',str(lower_list))

if __name__ == '__main__':
    # show_cycle_effeciency(use_rank=True)
    # show_cycle_effeciency(use_rank=False)
    # show_dmp_effeciency(test_id=109,flag=50)
    # show_dmp_effeciency(test_id=110,flag=50)
    # show_dmp_effeciency(test_id=111,flag=100)
    # show_dmp_effeciency(test_id=112,flag=50)

    # show_dmp_effeciency(test_id=127,flag=50)
    # show_dmp_effeciency(test_id=128,flag=50)
    # show_dmp_effeciency(test_id=129,flag=50)
    # show_dmp_effeciency(test_id=130,flag=50)
    # show_dmp_effeciency(test_id=131,flag=50)
    show_dmp_effeciency(test_id=132,flag=50)
    # show_dmp_effeciency(test_id=113,flag=50)
