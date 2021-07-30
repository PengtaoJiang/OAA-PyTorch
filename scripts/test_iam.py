import sys
import os
sys.path.append(os.getcwd())

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.LoadData import test_data_loader
from utils.Restore import restore
import matplotlib.pyplot as plt
from models import vgg1
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    
def get_arguments():
    parser = argparse.ArgumentParser(description='OAA')
    parser.add_argument("--root_dir", type=str, default='./')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default='')
    parser.add_argument("--test_list", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--arch", type=str,default='vgg_v0')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--num_workers", type=int, default=20)

    return parser.parse_args()

def get_model(args):
    model = vgg1.vgg16(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    pretrained_dict = torch.load(args.restore_from)['state_dict']
    model_dict = model.state_dict()
    
    print(model_dict.keys())
    print(pretrained_dict.keys())
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return  model

def validate(args):
    print('\nvalidating ... ', flush=True, end='')

    model = get_model(args)
    model.eval()
    val_loader = test_data_loader(args)
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, img, label_in = dat
            label = label_in.cuda(non_blocking=True)
            logits = model(img)

            cv_im = cv2.imread(img_name[0])
            cv_im_gray = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
            height, width = cv_im.shape[:2]

            for l, featmap in enumerate(logits):
                maps = featmap.cpu().data.numpy()
                im_name = args.save_dir + img_name[0].split('/')[-1][:-4]
                labels = label_in.long().numpy()[0]
                for i in range(int(args.num_classes)):
                    if labels[i] == 1:
                        att = maps[i]
                        att[att < 0] = 0
                        att = att / (np.max(att) + 1e-8)
                        att = np.array(att * 255, dtype=np.uint8)
                        out_name = im_name + '_{}.png'.format(i)
                        att = cv2.resize(att, (width, height), interpolation=cv2.INTER_CUBIC)
                        #att = cv_im_gray * 0.2 + att * 0.8
                        cv2.imwrite(out_name, att)
                        #plt.imsave(out_name, att, cmap=colormap(i))

if __name__ == '__main__':
    args = get_arguments()
    validate(args)
