import os
import sys
import time
import signal
import importlib
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision
import re
import torch.nn.functional as F

from .utils import *
from .models.multi_column import MultiColumn
from .transforms_video import *
from .models.model3D_1 import Model

class Base_eval:
    def __init__(self,opt=None):
        self.opt = opt
        # self.root = '/scr1/system/beta-robot/base_eval'
        # self.root = os.path.join(os.getcwd(),'base_eval')
        self.root = os.path.join(self.opt.project_root,'scripts','Eval','base_eval')

        self.config = load_json_config (os.path.join(self.root,"configs/pretrained/config_model1_left_right.json"))
        
        # set column model
        # file_name = self.config['conv_model']
        # self.cnn_def = importlib.import_module ("{}".format (file_name))
        
        # setup device - CPU or GPU
        self.device = torch.device ("cuda")
        self.device_ids = [0]



    def videoloader(self,filepath):
        transform_pre = ComposeMix ([
            [Scale (int (1.4 * self.config['input_spatial_size'])), "img"],
            [torchvision.transforms.ToPILImage (), "img"],
            [torchvision.transforms.CenterCrop (self.config['input_spatial_size']), "img"],
        ])
    
        transform_post = ComposeMix ([
            [torchvision.transforms.ToTensor (), "img"],
            [torchvision.transforms.Normalize (
                mean=[0.485, 0.456, 0.406],  # default values for imagenet
                std=[0.229, 0.224, 0.225]), "img"]
        ])
    
        imgs = []
        for file in os.listdir (filepath):
            tmp = cv2.imread (os.path.join (filepath, file))
            tmp = cv2.resize (tmp, (84, 84))
            imgs.append (tmp)
    
        imgs = transform_pre (imgs)
        imgs = transform_post (imgs)
    
        num_frames = len (imgs)
        num_frames_necessary = 72
    
        if len (imgs) < 72:
            imgs.extend ([imgs[-1]] * (72 - len (imgs)))
    
        data = torch.stack (imgs)
        data = data.permute (1, 0, 2, 3)
        return data


    def get_reward (self,filepath):
        model_name = "model3D_1"
        # create model
        # model = MultiColumn (self.config["num_classes"], self.cnn_def.Model, int (self.config["column_units"]))
        model = MultiColumn (self.config["num_classes"], Model, int (self.config["column_units"]))
        # multi GPU setting
        model = torch.nn.DataParallel (model, self.device_ids).to (self.device)
    
        save_dir = os.path.join (os.path.join(self.root,"trained_models/pretrained/" + self.config['model_name']))
        checkpoint_path = os.path.join (save_dir, 'model_best.pth.tar')
    
        checkpoint = torch.load (checkpoint_path)
        model.load_state_dict (checkpoint['state_dict'])
        model.eval ()
    
        logits_matrix = []
        features_matrix = []
        targets_list = []
        item_id_list = []
    
        with torch.no_grad ():
            input = self.videoloader (filepath)
            input = input.float ().unsqueeze (0)
            input_var = [input.to (self.device)]
            output = model (input_var, False)
            output = F.softmax (output, 1)
            output = output.cpu ().detach ().numpy ()
            output = np.squeeze (output)
    
            output_index = np.argsort (output * -1.0)
    
            return (output, output_index)


if __name__ == "__main__":
    agent = Base_eval()
    filepath = '/scr1/system/beta-robot/dataset/actions/106-2/frames'
    output,output_index = agent.get_reward (filepath)
    print(output_index)
