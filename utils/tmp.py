import os
import shutil
import random

def sample_imgs(dir):
    target_dir = './small/{}'.format(dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for file in os.listdir(dir):
        p = random.random()
        if p<0.35:
            shutil.copy(os.path.join(dir,file),os.path.join(target_dir,file))
            print(file)

sample_imgs('./trainA')
sample_imgs('./trainB')
