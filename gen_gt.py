
#######################################################################
# This file is provided by Peng-Tao Jiang. If you have any questions, #
# please feel free to contact me (pt.jiang@mail.nankai.edu.cn).       #
#######################################################################
import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

# set path for data
data_path = '/home/ubuntu/Project/datasets/VOCdevkit/VOC2012/'
train_lst_path = data_path + 'ImageSets/Segmentation/train_cls.txt'
im_path = data_path + 'JPEGImages/'
sal_path = data_path + 'saliency_aug/'
att_path = './runs/exp1/attention/'
save_path = './runs/exp1/proxy-gt/'

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

# generate proxy ground-truth
def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    im_name = im_path + name + '.jpg'
    bg_name = sal_path + name + '.png'
    #print(bg_name)
    if not os.path.exists(bg_name):
        return
    
    sal = cv2.imread(bg_name, 0)
    height, width = sal.shape
    gt = np.zeros((21, height, width), dtype=np.float32)
    sal = np.array(sal, dtype=np.float32)
    
    # some thresholds. 
    conflict = 0.9
    fg_thr = 0.3
    # the below two values are used for generating uncertainty pixels
    bg_thr = 32
    att_thr = 0.8

    # use saliency map to provide background cues
    gt[0] = (1 - (sal / 255))
    init_gt = np.zeros((height, width), dtype=float) 
    sal_att = sal.copy()  

    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue
        
        # normalize attention to [0, 1] 
        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        gt[cls+1] = att.copy()
        sal_att = np.maximum(sal_att, (att > att_thr) *255)
    
    
    # throw low confidence values for all classes
    gt[gt < fg_thr] = 0
    
    # conflict pixels with multiple confidence values
    bg = np.array(gt > conflict, dtype=np.uint8)  
    bg = np.sum(bg, axis=0)
    gt = gt.argmax(0).astype(np.uint8)
    gt[bg > 1] = 255

    # pixels regarded as background but confidence saliency values 
    bg = np.array(sal_att >= bg_thr, dtype=np.uint8) * np.array(gt == 0, dtype=np.uint8)
    gt[bg > 0] = 255  

    # this is an engineering idea, for an image with a small ratio of semantic objects,
    # we ignore the whole image, I find that this operation have little impact on 
    out = gt.copy() 
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        out[...] = 255

    # output the proxy labels using the VOC12 label format
    out = Image.fromarray(out.astype(np.uint8), mode='P')
    #out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)
    
### Parallel Mode
pool = multiprocessing.Pool(processes=16)
pool.map(gen_gt, range(len(lines)))
# pool.map(gen_gt, range(50))
pool.close()
pool.join()

# Loop Mode
#for i in range(len(lines)):
#    gen_gt(i)
