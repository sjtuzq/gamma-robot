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

if __name__ == '__main__':
    from utils import *
    from models.multi_column import MultiColumn
    from transforms_video import *
    from models.model3D_1 import Model
else:
    from .utils import *
    from .models.multi_column import MultiColumn
    from .transforms_video import *
    from .models.model3D_1 import Model

class Base_eval:
    def __init__(self,opt=None):
        self.opt = opt
        # self.root = '/scr1/system/beta-robot/base_eval'
        # self.root = os.path.join(os.getcwd(),'base_eval')
        if __name__ == '__main__':
            self.root = '/scr1/system/gamma-robot/scripts/Eval/base_eval'
        else:
            self.root = os.path.join(self.opt.project_root,'scripts','Eval','base_eval')

        self.config = load_json_config (os.path.join(self.root,"configs/pretrained/config_model1_left_right.json"))
        
        # set column model
        # file_name = self.config['conv_model']
        # self.cnn_def = importlib.import_module ("{}".format (file_name))
        
        # setup device - CPU or GPU
        self.device = torch.device ("cuda")
        self.device_ids = [0]

        model_name = "model3D_1"
        # create model
        # model = MultiColumn (self.config["num_classes"], self.cnn_def.Model, int (self.config["column_units"]))
        self.model = MultiColumn (self.config["num_classes"], Model, int (self.config["column_units"]))
        # multi GPU setting
        self.model = torch.nn.DataParallel (self.model, self.device_ids).to (self.device)

        save_dir = os.path.join (os.path.join (self.root, "trained_models/pretrained/" + self.config['model_name']))
        checkpoint_path = os.path.join (save_dir, 'model_best.pth.tar')

        checkpoint = torch.load (checkpoint_path)
        self.model.load_state_dict (checkpoint['state_dict'])
        self.model.eval ()



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
    
        # num_frames = len (imgs)
        # num_frames_necessary = 72
    
        if len (imgs) < 72:
            imgs.extend ([imgs[-1]] * (72 - len (imgs)))
    
        data = torch.stack (imgs)
        data = data.permute (1, 0, 2, 3)
        return data


    def get_baseline_reward (self,filepath):
        with torch.no_grad ():
            input = self.videoloader (filepath)
            input = input.float ().unsqueeze (0)
            input_var = [input.to (self.device)]
            output = self.model (input_var, False)
            output = F.softmax (output, 1)
            output = output.cpu ().detach ().numpy ()
            output = np.squeeze (output)
    
            output_index = np.argsort (output * -1.0)
    
            return (output, output_index)


if __name__ == "__main__":
    agent = Base_eval()
    i = 4
    filepath = '/scr1/system/gamma-robot/logs/td3_log/test1138/epoch-{}'.format (i)
    output, output_index = agent.get_baseline_reward (filepath)
    print (i, np.where (output_index == 9)[0][0],output[9]*173)

    # for i in range(100):
    #     filepath = '/scr1/system/gamma-robot/logs/td3_log/test135/epoch-{}'.format(i)
    #     output,output_index = agent.get_reward (filepath)
    #     print(output[9]*173)
        # print(np.where(output_index==9)[0][0])
    # print(output_index)
