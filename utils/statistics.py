import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_end_pos(test_id=92,trained=True,flag=50):
    project_root = '/scr1/system/gamma-robot/'
    log_dir = os.path.join(project_root,'logs','td3_log','test{}'.format(test_id))

    pos_data = []
    reward_data = []
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
            pos_data.append(list(pos))
            reward_data.append(reward)

    pos_data = np.array(pos_data)
    reward_data = np.array(reward_data)

    reward_data = reward_data-reward_data.mean()
    rank = np.argsort(reward_data)
    C = []
    for item in rank:
        if item<10:
            C.append('r')
        else:
            C.append('g')

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pos_data[:,0],pos_data[:,1],pos_data[:,2],c=C)

    plt.show()


# show_end_pos(test_id=90,trained=False,flag=50)
# show_end_pos(test_id=90,trained=True,flag=50)
#
# show_end_pos(test_id=91,trained=False,flag=100)
# show_end_pos(test_id=91,trained=True,flag=100)
#
# show_end_pos(test_id=92,trained=False,flag=50)
# show_end_pos(test_id=92,trained=True,flag=50)

show_end_pos(test_id=103,trained=False,flag=50)
show_end_pos(test_id=103,trained=True,flag=50)