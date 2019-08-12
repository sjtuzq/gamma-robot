from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def reverse(img):
    new_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = img[i][img.shape[1]-j-1]
    return new_img

def show_rgb():
    inter_n = 100

    world_data = np.load('./logs/reward_data_31_world_1.npy')
    camera_data = np.load('./logs/reward_data_31_camera_1.npy')

    # linear_img = np.zeros_like(camera_data)
    # for i in range(int(inter_n+1)):
    #     for j in range(int(inter_n+1)):
    #         linear_img[i][j] = i/inter_n*camera_data[inter_n][0]+j/inter_n*camera_data[0][inter_n]
    #
    # img = reverse(camera_data)
    #
    # for i in range (3):
    #     img[:, :, i] = (img[:, :, i] - img[:, :, i].min ())
    #     img[:,:,i] = img[:,:,i]* 1 / img[:, :, i].max ()
    # plt.cla ()
    # plt.imshow (img)
    # plt.show ()
    #
    # img = reverse(linear_img)
    # for i in range (3):
    #     img[:, :, i] = (img[:, :, i] - img[:, :, i].min ())
    #     img[:,:,i] = img[:,:,i]* 1 / img[:, :, i].max ()
    # plt.cla ()
    # plt.imshow (img)
    # plt.show ()
    #
    #
    # fig = plt.figure ()
    # ax = Axes3D (fig)
    # ax.view_init(elev=0,azim=0)
    # X = np.arange (0, 1+1/inter_n, 1/inter_n)
    # Y = np.arange (0, 1+1/inter_n, 1/inter_n)
    # X, Y = np.meshgrid (X, Y)
    # Z = camera_data[:,:,0]
    # ax.plot_surface (X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # plt.show ()

    plt.cla()
    fig = plt.figure ()
    ax = Axes3D (fig)
    # ax.view_init(elev=45,azim=0)
    # ax.view_init(elev=45,azim=90)
    ax.view_init(elev=0,azim=0)
    X = np.arange (0, 1+1/inter_n, 1/inter_n)
    Y = np.arange (0, 1+1/inter_n, 1/inter_n)
    X, Y = np.meshgrid (X, Y)
    Z = camera_data[:,:,1]
    ax.plot_surface (X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


if __name__ == '__main__':
    show_rgb()
