import argparse
import sys, os, gc, resource
import numpy as np
from PIL import Image
import cv2
import random

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as torch_data

from engine_fuse import Engine


parser = argparse.ArgumentParser(description='seg2D Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


from seg_fuse_suncg import Seg2DNet, PCA_Jittering


class TrainDataLoader(torch_data.Dataset):
    def __init__(self, path, train, img_transform = None, label_transform = None, num_classes = 12):

        super(TrainDataLoader, self).__init__()

        if train == True:
            HHA_path = '/media/jason/JetsonSSD/nyu_selected_HHA'
        else:
            HHA_path = '/media/jason/JetsonSSD/nyu_selected_val_HHA'

        fid = open(path, "r")
        self.colorlist = []
        self.categlist = []
        self.depthlist = []
        ind = 1
        for line in fid.readlines():
            line = line.rstrip("\n")
            line1 = line[0:len(line)-9] + 'category_suncg.png'
            if os.path.exists(line) and os.path.exists(line1):
                self.depthlist.append(HHA_path+'/%06d.png'%ind)
                self.colorlist.append(line)
                self.categlist.append(line1)
            ind = ind + 1
        fid.close()

        self.num_classes = num_classes
        self.color_transform = Compose([
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        self.depth_transform = Compose([
            ToTensor(),
            Normalize([.5282, .3914, .4266], [.1945, .2480, .1506])
        ])
        self.label_transform = label_transform

        self.resize_size = (384, 288) # 12:9

    def __len__(self):

        return len(self.colorlist)

    def __getitem__(self, index):

        depth = Image.open(self.depthlist[index]).convert('RGB')
        depth = depth.resize(self.resize_size, Image.ANTIALIAS)
        depth1 = depth.transpose(Image.FLIP_LEFT_RIGHT)
        depth = self.depth_transform(depth)
        depth1 = self.depth_transform(depth1)
        depth = torch.cat((depth, depth1), 0)

        color = Image.open(self.colorlist[index]).convert('RGB')
        color = color.resize(self.resize_size, Image.ANTIALIAS)
        color1 = color.transpose(Image.FLIP_LEFT_RIGHT)
        color = PCA_Jittering(color)
        color1 = PCA_Jittering(color1)
        color = self.color_transform(color)
        color1 = self.color_transform(color1)
        color = torch.cat((color, color1), 0)

        categ1 = cv2.imread(self.categlist[index], -1).astype(int)
        categ1 = cv2.resize(categ1, dsize=self.resize_size, interpolation=cv2.INTER_NEAREST)
        categ11 = np.fliplr(categ1)
        categ1 = torch.from_numpy(categ1.astype(np.int64))
        categ11 = torch.from_numpy(categ11.astype(np.int64))
        label1 = torch.cat((categ1, categ11), 0)

        return color, depth, label1



# CUDA_VISIBLE_DEVICES=0 python seg_fuse_nyu.py /home/jason/NYUv2/NYU_images 2>&1 | tee logs/seg_fuse_nyu.log
def main():
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = TrainDataLoader(os.path.join(args.data, 'nyu_train.txt'), train = True)
    val_dataset = TrainDataLoader(os.path.join(args.data, 'nyu_test.txt'), train = False)

    # load model
    model = Seg2DNet('./save_models/seg_RGB_suncg/checkpoint.pth.tar',
        './save_models/seg_depth_suncg/checkpoint.pth.tar',
        model=models.resnet101(False), num_classes=12)

    chpo = torch.load('./save_models/seg_fuse_suncg/checkpoint.pth.tar')
    model.load_state_dict(chpo['state_dict'])
    print "=> loaded checkpoint '{}'".format('./save_models/seg_fuse_suncg/checkpoint.pth.tar')

    # define loss function (criterion)
    cri_weights = torch.FloatTensor([0.0001, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])
    criterion = nn.CrossEntropyLoss(weight = cri_weights/torch.sum(cri_weights))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'workers': args.workers, 'start_epoch': args.start_epoch,
             'max_epochs': args.epochs, 'evaluate': args.evaluate, 'resume': args.resume,
             'multi_gpu': False, 'device_ids': [0, 1], 'use_gpu': use_gpu,
             'save_iter': 0, 'print_freq': args.print_freq, 'epoch_step': [20,40,60]}
    state['save_model_path'] = './save_models/seg_fuse_nyu'

    engine = Engine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':

    resource.setrlimit(resource.RLIMIT_STACK, (-1,-1))
    sys.setrecursionlimit(100000)

    main()

