import os
import shutil
import random

class Sthvideo:
    def __init__(self,id=94,output_path=None):
        self.id = id
        self.data_root = '/scr1/workspace/dataset/sth-sth'
        self.output = output_path
        self.train_file = os.path.join(self.data_root,'captions','train_videofolder.txt')
        self.val_file = os.path.join(self.data_root,'captions','val_videofolder.txt')
        video_by_id_data = {}
        with open(self.train_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                try:
                    video_by_id_data[line[2]].append(line)
                except:
                    video_by_id_data[line[2]] = [line]
        self.video_by_id_data = video_by_id_data

    def get_video_by_id(self,id=94):
        return self.video_by_id_data[str(id)]

    def move_video_by_id(self,id=94):
        target_folder = os.path.join(self.output,'videos',str(id))
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        for item in self.video_by_id_data[str(id)]:
            video = os.path.join(self.data_root,'videos','%d.webm'%int(item[0]))
            shutil.copy(video,target_folder)

    def move_frame_by_id(self,id=94):
        for item in self.video_by_id_data[str(id)]:
            frame_folder = os.path.join(self.data_root,'extract_frames','%d'%int(item[0]))
            p = random.random()
            if p<0.03:
                for frame in os.listdir(frame_folder):
                    p = random.random()
                    shutil.copy(os.path.join(frame_folder,frame),self.output)
                    os.rename(os.path.join(self.output,frame),os.path.join(self.output,item[0]+'_'+frame))

if __name__ == '__main__':
    item = Sthvideo()
    item.move_frame_by_id()
