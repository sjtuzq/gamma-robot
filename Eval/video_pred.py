import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import datasets_video
import pdb
from torch.nn import functional as F
import os

def get_pred(video_path,caption_path,opt):
    # options
    parser = argparse.ArgumentParser(
        description="TRN testing on the full validation set")
    # parser.add_argument('dataset', type=str, choices=['something','jester','moments','charades'])
    # parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])

    parser.add_argument('--dataset', type=str, default='somethingv2')
    parser.add_argument('--modality', type=str, default='RGB')

    parser.add_argument('--weights', type=str,default='model/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--save_scores', type=str, default=None)
    parser.add_argument('--test_segments', type=int, default=8)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=10)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='TRNmultiscale',
                        choices=['avg', 'TRN','TRNmultiscale'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--img_feature_dim',type=int, default=256)
    parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
    parser.add_argument('--softmax', type=int, default=0)

    args = parser.parse_args()


    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        prob, pred = output.topk(maxk, 1, True, True)
        prob = prob.t().data.numpy().squeeze()
        pred = pred.t().data.numpy().squeeze()
        return prob,pred

    categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality,opt)
    num_class = len(categories)

    net = TSN(num_class, args.test_segments if args.crop_fusion_type in ['TRN','TRNmultiscale'] else 1, args.modality,
              base_model=args.arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              opt = opt
              )

    try:
        checkpoint = torch.load(args.weights)
    except:
        args.weights = os.path.join(opt.project_root,'scripts/Eval/',args.weights)
        checkpoint = torch.load (args.weights)

    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(video_path,
                       caption_path,
                       num_segments=args.test_segments,
                       new_length=1 if args.modality == "RGB" else 5,
                       modality=args.modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                           GroupNormalize(net.input_mean, net.input_std),
                       ])),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    #net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    data_gen = enumerate(data_loader)

    output = []


    def eval_video(video_data):
        i, data, label = video_data
        num_crop = args.test_crops

        if args.modality == 'RGB':
            length = 3
        elif args.modality == 'Flow':
            length = 10
        elif args.modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+args.modality)

        input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                            volatile=True)
        rst = net(input_var)
        if args.softmax==1:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst)

        rst = rst.data.cpu().numpy().copy()

        if args.crop_fusion_type in ['TRN','TRNmultiscale']:
            rst = rst.reshape(-1, 1, num_class)
        else:
            rst = rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

        return i, rst, label[0]

    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

    prob_all, pred_all = [],[]
    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        rst = eval_video((i, data, label))
        output.append(rst[1:])
        prob, pred = accuracy(torch.from_numpy(np.mean(rst[1], axis=0)), label, topk=(1, 174))
        prob_all.append(prob)
        pred_all.append(pred)
    return prob_all,pred_all

def run(video_path,caption_path,result_path):
    prob,pred = get_pred(video_path,caption_path)
    result_file = open(result_path,'w')
    with open(caption_path,'r') as caption_file:
        for line_id, line in enumerate(caption_file.readlines()):
            result_file.write(line)
            result_file.write(str(list(pred[line_id])))
            rank = np.argwhere(pred[line_id]==int(line.split()[-1]))[0][0]+1
            result_file.write('\nranking: {}'.format(rank))
            result_file.write('\n\n')
    result_file.close()
    caption_file.close()


if __name__ == '__main__':
    # video_path = '/scr1/system/alpha-robot/dataset/cycled_video/cycled_img/cycled_data'
    # caption_path = '/scr1/system/alpha-robot/dataset/cycled_video/cycled_img/test_input.txt'
    # result_path = '/scr1/system/alpha-robot/dataset/cycled_video/cycled_img/test_output.txt'

    # video_path = '/scr1/system/alpha-robot/dataset/cycled_video/robot_video/frames'
    # caption_path = '/scr1/system/alpha-robot/dataset/cycled_video/robot_video/caption.txt'
    # result_path = '/scr1/system/alpha-robot/dataset/cycled_video/robot_video/result.txt'
    # run(video_path, caption_path, result_path)

    video_path = '/scr1/system/beta-robot/dataset/own_robot_video/masked_frames'
    caption_path = '/scr1/system/beta-robot/dataset/own_robot_video/exploring/caption_test.txt'
    result_path = '/scr1/system/beta-robot/dataset/own_robot_video/exploring/masked_result_test.txt'
    run(video_path,caption_path,result_path)