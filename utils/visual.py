
from images2gif import writeGif
from PIL import Image
import os



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





if __name__ == '__main__':
    test_id = 65
    epoch_id = 800
    get_gif(test_id,epoch_id)