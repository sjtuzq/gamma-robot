
from images2gif import writeGif
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


from tensorboardX import SummaryWriter

def get_gif(test_id,epoch_id):
    log_root = '/src1/system/gamma-robot/logs/td3_log/'
    log_dir = os.path.join(log_root,'test{}/epoch-{}'.format(test_id,epoch_id))
    cmd = 'convert -delay 120 -loop 0 {}/*.jpg {}/video.gif'.format(log_dir,log_dir)
    os.system(cmd)

    log_dir = '/scr1/system/gamma-robot/logs/td3_log/test65/epoch-800'
    file_names = sorted ( (fn for fn in os.listdir (log_dir) if fn.endswith ('.jpg')))

    images = [Image.open (os.path.join(log_dir,fn)) for fn in file_names]

    # size = (150, 150)
    # for im in images:
    #     im.thumbnail (size, Image.ANTIALIAS)

    filename = os.path.join(log_dir,"my_gif.gif")
    writeGif (filename, images, duration=0.2)


def smooth_show():
    writer = SummaryWriter('./logs')
    log_85 = '/scr1/system/gamma-robot/logs/tmp/log_85.npy'
    log_87 = '/scr1/system/gamma-robot/logs/tmp/log_87.npy'
    log85_data = np.load(log_85)
    log87_data = np.load(log_87)
    # print(log85_data)

    for i,item in enumerate(log85_data):
        writer.add_scalar('log85',item,i)

    for i,item in enumerate(log87_data):
        writer.add_scalar('log87',item,i)

    # xnew = np.linspace(0,log85_data.shape[0],10000)
    # power_smooth = spline(range(log85_data.shape[0]),log85_data,xnew)
    # plt.plot(xnew,power_smooth)
    # plt.show()

    # gap = 100
    # log85_ave_data = []
    # for i in range(int(log85_data.shape[0]/gap)):
    #     tmp = log85_data[i*gap:i*gap+gap].sum()/gap
    #     log85_ave_data.append(tmp)
    # plt.plot(log85_ave_data)
    # plt.show()

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

class Visual:
    def __init__(self,test_id=0,epoch_id=0):
        self.test_id = test_id
        self.epoch_id = epoch_id
        self.log_file = '/scr1/system/gamma-robot/logs/td3_log/test{}/epoch-{}.txt'\
            .format(self.test_id,self.epoch_id)
        self.traj_folder = safe_path('/scr1/system/gamma-robot/logs/td3_log/test{}/traj/'.format(self.test_id))

    def update(self,epoch_id):
        self.epoch_id = epoch_id
        self.log_file = '/scr1/system/gamma-robot/logs/td3_log/test{}/epoch-{}.txt' \
            .format (self.test_id, self.epoch_id)

    def get_xyz(self):
        self.points = []
        with open(self.log_file,'r') as reader:
            for line in reader.readlines():
                line = line.strip().split(':')
                if line[0]=='now_pos':
                    self.points.append(eval(line[1]))
        self.points = np.array(self.points)

    def show_xyz(self):
        self.get_xyz ()
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure ()
        ax = fig.gca (projection='3d')
        x = self.points[:,0]
        y = self.points[:,1]
        z = self.points[:,2]
        ax.plot (x, y, z, label='parametric curve')
        ax.legend ()

        # parser.add_argument ('--axis_limit_x', default='[0.04,0.6]', type=str)  #
        # parser.add_argument ('--axis_limit_y', default='[-0.40,0.25]', type=str)  #
        # parser.add_argument ('--axis_limit_z', default='[0.26,0.7]', type=str)  #

        plt.xlim ((0.04,0.6))
        # plt.xlim (bottom=0.6)

        plt.ylim ((-0.4,0.25))
        # plt.ylim (bottom=-0.40)

        ax.set_zlim ((0.26,0.7))
        # plt.ylim (bottom=0.7)

        plt.savefig(os.path.join(self.traj_folder,'trajectory-{}.jpg'.format(self.epoch_id)))
        plt.cla()


if __name__ == '__main__':
    agent = Visual (test_id=91)
    # agent.update(95)
    # agent.show_xyz()
    for epoch in range (300):
        try:
            agent.update (epoch + 1)
            agent.show_xyz ()
        except:
            print(epoch)
            continue