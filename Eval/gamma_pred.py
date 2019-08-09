import os
import numpy as np
import shutil
import time
import argparse
import torch
import random
import torch.nn.functional as F

from video_pred import get_pred
from base_eval.base_pred import Base_eval


def safe_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

class Frame_eval:
    def __init__(self,img_path=None,frame_len=20,start_id=0,memory_path=None,class_label=45,opt=None):
        self.img_path = img_path
        self.frame_len = frame_len
        self.start_id = start_id
        self.memory_path = safe_path(memory_path)
        self.class_label = class_label
        self.video_path = safe_path(os.path.join(self.memory_path,'videos'))
        self.video_id = len(os.listdir(self.video_path))+1
        self.tmp_dir = safe_path(os.path.join(self.video_path,'{}'.format(self.video_id)))
        self.opt = opt
        self.base_eval = Base_eval(self.opt)

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
        if self.opt.use_embedding:
            self.class_label = self.opt.load_embedding

        if self.opt.use_trn:
            prob, pred = get_pred (self.video_path, self.caption_file, self.opt)
            prob,pred = prob[0],pred[0]
            trn_rank = np.argwhere (pred == self.class_label).squeeze () + 1
            trn_reward = prob[np.argwhere (pred == self.class_label).squeeze ()]
            return trn_rank,trn_reward

        else:
            output, output_index = self.base_eval.get_baseline_reward(self.img_path)
            rank = np.argwhere (output_index == self.class_label).squeeze () + 1
            reward = output[self.class_label] * 173
            reward = F.sigmoid(torch.tensor(reward).float())- F.sigmoid(torch.tensor(1).float())

            # the following function does not work
            # reward = output[self.class_label] * 173 - 1
            # reward = F.sigmoid (torch.tensor (reward).float ()) - 0.5

            if self.opt.use_embedding:
                # reward1 = F.sigmoid (torch.tensor (output[self.opt.embedding_list[0]] * 173).float ()) - F.sigmoid (torch.tensor (1).float ())
                # reward2 = F.sigmoid (torch.tensor (output[self.opt.embedding_list[1]] * 173).float ()) - F.sigmoid (torch.tensor (1).float ())
                # reward = [reward1,reward2]
                # reward = [reward1*abs(reward1-reward2),reward2*abs(reward2-reward1)]
                reward = []
                for i in range(self.opt.embedding_dim):
                    action_reward = F.sigmoid (torch.tensor (output[self.opt.embedding_list[i]] * 173).float ()) \
                                    - F.sigmoid (torch.tensor (1).float ())
                    reward.append(action_reward)
                reward_mean = float(sum(reward))/self.opt.embedding_dim
                for i in range (self.opt.embedding_dim):
                    reward[i] = reward[i]*abs(reward[i]-reward_mean)*self.opt.embedding_dim
            return rank,reward

    # def eval(self):
    #     prob,pred = get_pred(self.video_path,self.caption_file,self.opt)
    #     prob,pred = prob[0],pred[0]
    #
    #     output, output_index = self.base_eval.get_baseline_reward(self.img_path)
    #
    #     # print(pred,output_index)
    #
    #     # using softmax
    #     if self.opt.prob_softmax:
    #         prob = torch.softmax(torch.tensor(prob).float(),0).data.numpy()
    #         prob = (prob - prob.mean())/np.std(prob)
    #
    #     # reward 1
    #     rank = np.argwhere (pred == self.class_label).squeeze () + 1
    #     reward1 = prob[np.argwhere (pred == self.class_label).squeeze ()]
    #
    #     # reward 2
    #     rank2 = np.argwhere (output_index == self.class_label).squeeze () + 1
    #     reward2 = output[self.class_label] * 173
    #
    #     reward3 = F.sigmoid(torch.tensor(reward2).float())- F.sigmoid(torch.tensor(1).float())
    #
    #     # # reward 3
    #     # reward3 = 173./rank
    #     #
    #     # # reward 4
    #     # reward4 = 173./rank2
    #
    #     return rank2,reward3

        # if self.class_label==107 and self.opt.merge_class and self.test_id==101:
        #     propability = 0
        #     # action_list = [104,105,106,107,108,109]
        #     # for action in action_list:
        #     #     rank_ = np.argwhere (pred == action).squeeze () + 1
        #     #     probability += prob[rank_ - 1]
        #
        #     rank_ = np.argwhere (pred[0] == self.class_label).squeeze () + 1
        #     probability = 173./prob[rank_ - 1]

        # if self.class_label==86:
        #     probability = prob[rank - 1] - prob[np.argwhere (pred == 45).squeeze ()]*0.8
        # if self.class_label==94:
        #     probability = prob[rank - 1] - prob[np.argwhere (pred == 45).squeeze ()]*0.8

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




def test_base_eval():
    agent = Base_eval()
    filepath = '/scr1/system/beta-robot/dataset/actions/106-2/frames'
    output,output_index = agent.get_reward (filepath)
    print(output_index)


if __name__ == '__main__':
    test_base_eval()
    # # img_path = '/scr1/system/beta-robot/dataset/actions/107-6/frames'
    # img_path = '/scr1/system/gamma-robot/logs/td3_log/test96/epoch-40'
    #
    # # writer = open('../tmp/test_gap.txt','w')
    # # frame_len_list = [10,20,30,40,50,60,70,80,90]
    # # for frame_len in frame_len_list:
    # agent = Frame_eval(img_path=img_path,
    #                    frame_len = 60,
    #                    start_id = 40,
    #                    memory_path = '/scr1/system/beta-robot/dataset/ddpg_log/memory',
    #                    class_label=107)
    # agent.update(start_id=0)
    # agent.get_caption()
    # rank,propability = agent.eval()
    # print(rank,propability)

