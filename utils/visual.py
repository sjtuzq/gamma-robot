
from images2gif import writeGif
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

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


if __name__ == '__main__':
    # test_id = 65
    # epoch_id = 800
    # get_gif(test_id,epoch_id)
    smooth_show()