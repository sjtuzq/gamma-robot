import os
import sys
import time
import signal
import importlib
import torch
import torch.nn as nn
import numpy as np
import cv2

from utils import *
from models.multi_column import MultiColumn
import torchvision
from transforms_video import *
import re
import torch.nn.functional as F


# np.set_printoptions (precision=4, suppress=True, linewidth=300)
# sys.path.insert (0, "/scr1/system/beta-robot/smth-smth-v2-baseline-with-models")

# load configurations
config = load_json_config ("/scr1/system/beta-robot/smth-smth-v2-baseline-with-models/configs/pretrained/config_model1_left_right.json")

# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module ("{}".format (file_name))

# setup device - CPU or GPU
device = torch.device ("cuda")
device_ids = [0]



def videoloader (filepath):
    transform_pre = ComposeMix ([
        [Scale (int (1.4 * config['input_spatial_size'])), "img"],
        [torchvision.transforms.ToPILImage (), "img"],
        [torchvision.transforms.CenterCrop (config['input_spatial_size']), "img"],
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


def get_reward (filepath):
    model_name = "model3D_1"
    # create model
    model = MultiColumn (config["num_classes"], cnn_def.Model, int (config["column_units"]))
    # multi GPU setting
    model = torch.nn.DataParallel (model, device_ids).to (device)

    save_dir = os.path.join ("/scr1/system/beta-robot/smth-smth-v2-baseline-with-models/trained_models/pretrained/" + config['model_name'])
    checkpoint_path = os.path.join (save_dir, 'model_best.pth.tar')

    checkpoint = torch.load (checkpoint_path)
    model.load_state_dict (checkpoint['state_dict'])
    model.eval ()

    logits_matrix = []
    features_matrix = []
    targets_list = []
    item_id_list = []

    with torch.no_grad ():
        input = videoloader (filepath)
        input = input.float ().unsqueeze (0)
        input_var = [input.to (device)]
        output = model (input_var, False)
        output = F.softmax (output, 1)
        output = output.cpu ().detach ().numpy ()
        output = np.squeeze (output)

        output_index = np.argsort (output * -1.0)

        return (output, output_index)


if __name__ == "__main__":
    filepath = '/scr1/system/beta-robot/dataset/actions/106-2/frames'
    output,output_index = get_reward (filepath)
    print(output_index)
