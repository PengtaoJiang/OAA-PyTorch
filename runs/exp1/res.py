import numpy as np
import cv2
import logging
import os 
from os.path import exists 
import matplotlib as mpl
import matplotlib.pyplot as plt

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    
def load_dataset(test_lst):
    logging.info('Beginning loading dataset...')
    im_lst = []
    label_lst = []
    with open(test_lst) as f:
        test_names = f.readlines()
    lines = open(test_lst).read().splitlines()
    for line in lines:
        fields = line.split()
        im_name = fields[0]
        im_labels = []
        for i in range(len(fields)-1):
            im_labels.append(int(fields[i+1]))
        im_lst.append(im_name)
        label_lst.append(im_labels)
    return im_lst, label_lst

if __name__ == '__main__':
    
    train_lst = '/home/miao/Projects/Classification/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt'
    root_folder = '/home/miao/Projects/Classification/data/VOCdevkit/VOC2012'
    im_lst, label_lst = load_dataset(train_lst)
    
    atten_path = 'accu_att'
    save_path = 'accu_att_zoom'
    if not exists(save_path):
        os.mkdir(save_path)
    for i in range(len(im_lst)):
        im_name = '{}/JPEGImages/{}.jpg'.format(root_folder, im_lst[i])
        im_labels = label_lst[i]
        
        img = cv2.imread(im_name)
        height, width = img.shape[:2]
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for label in im_labels:
            att_name = '{}/{}_{}.png'.format(atten_path, i, label)
            if not exists(att_name):
                continue 
            att = cv2.imread(att_name, 0)


            att = cv2.resize(att, (width, height), interpolation=cv2.INTER_CUBIC)
            min_value = np.min(att)
            max_value = np.max(att)
            att = (att - min_value) / (max_value - min_value + 1e-8)
            att = np.array(att*255, dtype = np.uint8)
            
            att = im_gray * 0.2 + att * 0.8
            save_name = '{}/{}_{}.png'.format(save_path, im_lst[i], label)
            #plt.imsave(save_name, att, cmap=plt.cm.jet)
            plt.imsave(save_name, att, cmap=colormap(label))
            #cv2.imwrite(save_name, att)


