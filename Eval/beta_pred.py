"""
    beta version video prediction module
"""
import os
import numpy as np
import shutil
import time
import argparse

from video_pred import get_pred

class Video_eval:
    def __init__(self,class_id=45,video_id=0,mode='mask'):
        self.video_root_path = '../../dataset/actions'
        self.class_id = class_id
        self.video_id = video_id
        self.mode = mode
        self.get_caption()

    def get_caption (self):
        self.action_path = os.path.join (self.video_root_path, '{}-{}'.format (self.class_id, self.video_id))
        self.video_dir = os.path.join(self.action_path,'videos')
        if not os.path.exists(self.video_dir):
            os.mkdir(self.video_dir)
        self.video_num = len(os.listdir(self.video_dir))
        self.video_path = os.path.join(self.video_dir,'{}'.format(self.video_num))
        if not os.path.exists(self.video_path):
            os.mkdir(self.video_path)

        if self.mode == 'real':
            self.tmp_video = os.path.join (self.action_path,'frames')
        elif self.mode == 'mask':
            self.tmp_video = os.path.join (self.action_path, 'masked_frames')
        else:
            self.tmp_video = os.path.join (self.action_path, 'simulation_frames')

        for file in os.listdir(self.tmp_video):
            shutil.copy(os.path.join(self.tmp_video,file),os.path.join(self.video_path,file))
        self.caption_path = os.path.join(self.action_path,'caption.txt')
        self.result_path = os.path.join(self.action_path,'result.txt')
        with open(self.caption_path,'w') as file:
            file.write('{} {} {}\n'.format(self.video_num,len(os.listdir(self.video_path)),self.class_id))
        file.close()

    def eval(self):
        prob,pred = get_pred(self.video_dir,self.caption_path)
        return prob,pred


    def pred(self):
        prob,pred = get_pred(self.video_dir,self.caption_path)
        result_file = open (self.result_path, 'w')
        with open (self.caption_path, 'r') as caption_file:
            for line_id, line in enumerate (caption_file.readlines ()):
                result_file.write (line)
                result_file.write (str (list (pred[line_id])))
                rank = np.argwhere (pred[line_id] == int (line.split ()[-1]))[0][0] + 1
                result_file.write ('\nranking: {}'.format (rank))
                result_file.write ('\n')
                result_file.write (str (list (prob[line_id])))
        result_file.close ()
        return prob,pred

def get_one_result():
    parser = argparse.ArgumentParser(description='simulation video generation')
    parser.add_argument('--class_id', default='1000', type=int, help='class id')
    parser.add_argument('--video_id', default='0', type=int, help='video id')
    parser.add_argument('--mode', default='sim', type=str, help='mode')
    opt = parser.parse_args()

    start = time.time()
    agent = Video_eval (opt.class_id, opt.video_id, opt.mode)
    prob, pred = agent.pred ()
    print(prob[0],pred[0])
    end = time.time()
    print('using time: {} s'.format(end-start))

def get_batch_result():
    video_list = [5, 8, 10, 12, 13, 15, 17, 27, 28, 40, 41, 42, 43, 44,
                  45, 47, 49, 55, 56, 85, 86, 87, 93, 94, 100, 101, 104, 105, 106, 107, 109]
    for video in video_list:
        for j in range(10):
            print(video,j)
            try:
                agent = Video_eval(video,j,'mask')
                prob,pred = agent.pred()
            except:
                print('error')

def collect_batch_result():
    action_path = '../../dataset/actions'
    output_path = '../../dataset/actions_result_batch.txt'
    output_writer = open(output_path,'w')
    video_list = [5, 8, 10, 12, 13, 15, 17, 27, 28, 40, 41, 42, 43, 44,
                  45, 47, 49,55,56,85,86,87,93,94,100,101,104,105,106,107,109]
    all_video_result = {}
    for video in video_list:
        now_video_result = []
        for j in range(10):
            try:
                result_file = os.path.join(action_path,'{}-{}/result.txt'.format(video,j))
                with open(result_file,'r') as f:
                    lines = f.readlines()[2]
                    lines = int(lines.strip().split(':')[-1])
                    now_video_result.append(lines)
            except:
                now_video_result.append(-1)
        all_video_result[video] = now_video_result
        output_writer.write('{}:\n{}   {}\n\n'.format(str(video),str(now_video_result),str(sum(now_video_result)/10)))

if __name__ == '__main__':
    # get_batch_result()
    # collect_batch_result()
    get_one_result()
