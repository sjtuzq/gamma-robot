import os
import numpy as np
import shutil
import time
import argparse
import torch

from video_pred import get_pred

def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

class Frame_eval:
    def __init__(self,img_path=None,frame_len=20,start_id=0,memory_path=None,class_label=45):
        self.img_path = img_path
        self.frame_len = frame_len
        self.start_id = start_id
        self.memory_path = safe_path(memory_path)
        self.class_label = class_label
        self.video_path = safe_path(os.path.join(self.memory_path,'videos'))
        self.video_id = len(os.listdir(self.video_path))+1
        self.tmp_dir = safe_path(os.path.join(self.video_path,'{}'.format(self.video_id)))

    def update(self,img_path=None,start_id=0):
        if img_path is not None:
            self.img_path = img_path
        self.start_id = start_id
        self.video_id = len (os.listdir (self.video_path)) + 1
        self.tmp_dir = safe_path (os.path.join (self.video_path, '{}'.format (self.video_id)))
        # self.get_caption ()

    def get_caption(self):
        for i in range(self.frame_len):
            img_file = os.path.join(self.img_path,'%06d.jpg' % (self.start_id+i))
            tgt_file = os.path.join(self.tmp_dir,'%06d.jpg'%i)
            try:
                shutil.copy(img_file,tgt_file)
            except:
                continue
        self.caption_file = os.path.join(self.memory_path,'caption.txt')
        with open(self.caption_file,'w') as f:
            f.write('{} {} {}\n'.format(self.video_id,len(os.listdir(self.tmp_dir)),self.class_label))
        f.close()
        self.result_file = os.path.join(self.memory_path,'result.txt')

    def eval(self):
        prob,pred = get_pred(self.video_path,self.caption_file)
        prob = prob[0]

        # using softmax
        # prob = torch.softmax(torch.tensor(prob).float(),0).data.numpy()

        rank = np.argwhere (pred[0] == self.class_label).squeeze () + 1
        probability = prob[rank - 1]
        if self.class_label==86:
            probability = prob[rank - 1] - prob[np.argwhere (pred[0] == 45).squeeze ()]*0.8
        if self.class_label==94:
            probability = prob[rank - 1] - prob[np.argwhere (pred[0] == 45).squeeze ()]*0.8
        return rank,probability

def test_class():
    agent = Frame_eval(img_path='../../../dataset/actions/43-0/simulation_frames',
                       frame_len = 20,
                       start_id = 0,
                       memory_path='../../../dataset/actions/43-0/memory',
                       class_label=43)

    rank,probability = agent.eval()
    print(rank,probability)

    img_path = '../../../dataset/actions/44-0/simulation_frames'
    agent.update(img_path,start_id=20)
    rank,probability = agent.eval()
    print(rank,probability)

def rename(data_root):
    # data_root = './simulation_frames'
    for file in os.listdir(data_root):
        file_id = int(file.split('-')[1].split('.')[0])
        new_name = os.path.join(data_root,'%06d.jpg'%file_id)
        os.rename(os.path.join(data_root,file),new_name)

def batch_get_name():
    writer = open('caption.txt','w')
    data_root = 'videos'
    for dir in os.listdir(data_root):
        now_root = os.path.join(data_root,dir)
        dir_num = len(os.listdir(now_root))
        for file in os.listdir(now_root):
            file_id = int (file.split ('-')[1].split ('.')[0])
            new_name = os.path.join (now_root, '%06d.jpg' % file_id)
            os.rename (os.path.join (now_root, file), new_name)
        new_dir_name = dir.split('-')[1]
        os.rename(now_root,os.path.join(data_root,new_dir_name))
        writer.write('{} {} {}\n'.format(int(new_dir_name),dir_num,45))

def rerank():
    reader = open('caption.txt','r')
    data = reader.readlines()
    data = sorted(data,key=lambda x:int(x.split()[0]))

    writer = open('caption2.txt','w')
    for item in data:
        writer.write(str(item))

def check_frames():
    img_path = '/scr1/system/beta-robot/dataset/epoch_log/test2/epoch-134'

    writer = open('../tmp/test_gap.txt','w')
    frame_len_list = [10,20,30,40,50,60,70,80,90]
    for frame_len in frame_len_list:
        agent = Frame_eval(img_path=img_path,
                           frame_len = frame_len,
                           start_id = 0,
                           memory_path = '/scr1/system/beta-robot/dataset/ddpg_log/memory',
                           class_label=45)
        # rename(img_path)

        rank_ave = 0.
        prob_ave = 0.
        id = 0.
        for i in range(10):
            agent.update(start_id=int(10-frame_len/10)*i)
            agent.get_caption ()
            rank, probability = agent.eval ()
            print (i, rank, probability)
            rank_ave += rank
            prob_ave += probability
            id += 1
            writer.write('frame_len:{}   rank:{}   prob:{}\n'.format(frame_len,rank,probability))
        rank_ave /= id
        prob_ave /= id
        writer.write ('average result: frame_len:{}   rank:{}   prob:{}\n\n'.format (frame_len, rank_ave, prob_ave))

    # for i in range(2192):
    #     agent.update(img_path='/scr1/system/beta-robot/dataset/epoch_log/test2/epoch-{}'.format(i))
    #     agent.get_caption()
    #     rank, probability = agent.eval ()
    #     print (i,rank, probability)


if __name__ == '__main__':
    img_path = '/scr1/system/beta-robot/dataset/ddpg_log/test14/epoch-2'

    # writer = open('../tmp/test_gap.txt','w')
    # frame_len_list = [10,20,30,40,50,60,70,80,90]
    # for frame_len in frame_len_list:
    agent = Frame_eval(img_path=img_path,
                       frame_len = 20,
                       start_id = 0,
                       memory_path = '/scr1/system/beta-robot/dataset/ddpg_log/memory',
                       class_label=86)
    agent.update(start_id=0)
    agent.get_caption()
    rank,propability = agent.eval()
    print(rank,propability)


