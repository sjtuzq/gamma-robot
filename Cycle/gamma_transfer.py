import os
import shutil
from video_train import train_model
from video_cycle import test_model

class Frame_transfer:
    def __init__(self,opt):
        self.opt = opt
        self.project_root = opt.project_root
        self.checkpoint_dir = os.path.join(self.project_root,'logs','cycle-data','checkpoint')
        self.train_dataset = os.path.join(self.project_root,'dataset')
        self.dataset_root = os.path.join(self.project_root,'logs','td3_log','test{}'.format(self.opt.test_id))
    def train(self):
        train_model(self.train_dataset,self.checkpoint_dir)

    def test(self,input_img,output_img):
        test_model(input_img,output_img,self.checkpoint_dir)

    def image_transfer(self,epoch_id):
        self.img_path = os.path.join(self.dataset_root,'epoch-{}'.format(epoch_id))
        self.cycle_memory = os.path.join(self.dataset_root,'memory','cycle-memory')
        if not os.path.exists(self.cycle_memory):
            os.mkdir(self.cycle_memory)
        test_a_path = os.path.join(self.cycle_memory,'testA')
        if not os.path.exists(test_a_path):
            os.mkdir(test_a_path)
        test_b_path = os.path.join(self.cycle_memory,'testB')
        if not os.path.exists(test_b_path):
            os.mkdir(test_b_path)
        for img in os.listdir(self.img_path):
            shutil.copy(os.path.join(self.img_path,img),os.path.join(self.cycle_memory,'testA',img))
        shutil.copy (os.path.join (self.img_path, img), os.path.join (self.cycle_memory, 'testB', img))
        self.test(self.cycle_memory,self.img_path)


if __name__ == '__main__':
    project_root = '/juno/u/qiangzhang/system/cycle-module'
    transfer_module = Frame_transfer(project_root)
    transfer_module.train()
