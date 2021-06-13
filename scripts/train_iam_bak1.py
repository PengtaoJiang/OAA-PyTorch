import sys
sys.path.append('/home/miao/Projects/OAA_pytorch/')

import torch
import argparse
import os
import time
import shutil
import json
import my_optim
import torch.optim as optim
from models import vgg1
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
from utils.LoadData import train_data_loader_iam
from tqdm import trange, tqdm

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print('Project Root Dir:', ROOT_DIR)

def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of OAA')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='None')
    parser.add_argument("--test_list", type=str, default='None')
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='61')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='')
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--att_dir", type=str, default='./runs/exp2/')

    return parser.parse_args()

class ExLoss(nn.Module):
    def __init__(self):
        super(ExLoss, self).__init__()

    def forward(self, input, target):
        assert(input.size() == target.size())
        scalar = torch.tensor([0]).float().cuda()
        pos = torch.gt(target, 0)
        neg = torch.eq(target, 0)
        pos_loss = -target[pos] * torch.log(torch.sigmoid(input[pos]))
        neg_loss = -torch.log(1-torch.sigmoid(input[neg]))
      
        loss = 0.0
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        if num_pos > 0:
            loss += 1.0 / num_pos.float() * torch.sum(pos_loss)
        if num_neg > 0:
            loss += 1.0 / num_neg.float() * torch.sum(neg_loss)
      
        return loss


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = vgg1.vgg16(pretrained=True, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2*args.lr},
        {'params': param_groups[2], 'lr': 10*args.lr},
        {'params': param_groups[3], 'lr': 20*args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    criterion = ExLoss()
    return  model, optimizer, criterion

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    train_loader = train_data_loader_iam(args)
    max_step = total_epoch*len(train_loader)
    args.max_step = max_step 
    print('Max step:', max_step)
    
    model, optimizer, criterion = get_model(args)
    print(model)
    model.train()
    end = time.time()

    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)
        
        for idx, dat in enumerate(train_loader):
            img, label = dat
            label = label.cuda(non_blocking=True)
            logits = model(img)

            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)
            loss_val = criterion(logits, label)
            
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            losses.update(loss_val.data.item(), img.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()
            
            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t' 
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, global_counter%len(train_loader), len(train_loader), 
                        optimizer.param_groups[0]['lr'], loss=losses))

        if current_epoch == args.epoch-1:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d.pth' %(args.dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
