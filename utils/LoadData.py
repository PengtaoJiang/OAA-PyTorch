# from torchvision import transforms
from .transforms import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from .imutils import ResizeShort
import os
from PIL import Image
import random

def train_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([#transforms.Resize(input_size),  
                                     ResizeShort(224),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([ResizeShort(224), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset(args.train_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_train, test=True)
    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)

    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

def test_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)

    tsfm_test = transforms.Compose([#transforms.Resize(input_size),  
                                     ResizeShort(224), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])  

    img_test = VOCDataset(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

def test_msf_data_loader(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_test = VOCDatasetMSF(args.test_list, root_dir=args.img_dir, num_classes=args.num_classes, scales=args.scales, transform=tsfm_test, test=True)
    val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return val_loader

class VOCDataset(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        if self.testing:
            return img_name, image, self.label_list[idx]
        
        return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels#, np.array(img_labels, dtype=np.float32)

####integral attention model learning######

def train_data_loader_iam(args, test_path=False, segmentation=False):
    if 'coco' in args.dataset:
        mean_vals = [0.471, 0.448, 0.408]
        std_vals = [0.234, 0.239, 0.242]
    else:
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose([ResizeShort(224),  
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    img_train = VOCDataset_iam(args.train_list, root_dir=args.img_dir, att_dir=args.att_dir, num_classes=args.num_classes, \
                    transform=tsfm_train, test=False)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader

class VOCDataset_iam(Dataset):
    def __init__(self, datalist_file, root_dir, att_dir, num_classes=20, transform=None, test=False):
        self.root_dir = root_dir
        self.att_dir = att_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list, self.label_name_list = \
                self.read_labeled_image_list(self.root_dir, self.att_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        im_labels = self.label_list[idx]
        im_label_names = self.label_name_list[idx]
        tmp = Image.open(im_label_names[0])
        h, w = tmp.size
        labels = np.zeros((self.num_classes, w, h), dtype=np.float32)

        for j in range(len(im_label_names)):
            label = im_labels[j]
            label_name = im_label_names[j]
            labels[label] = np.asarray(Image.open(label_name))
        labels /= 255.0

        if self.transform is not None:
            image = self.transform(image)
        
        return image, labels

    def read_labeled_image_list(self, data_dir, att_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()

        img_name_list = []
        label_list = []
        label_name_list = []

        for i, line in enumerate(lines):
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            img_name_list.append(os.path.join(data_dir, image))
            
            im_labels = []
            im_label_names = []

            for j in range(len(fields)-1):
                im_labels.append(int(fields[j+1]))    
                index = '{}_{}.png'.format(i, fields[j+1])
                im_label_names.append(os.path.join(att_dir, index))

            label_list.append(im_labels)
            label_name_list.append(im_label_names)

        return img_name_list, label_list, label_name_list

class VOCDatasetMSF(Dataset):
    def __init__(self, datalist_file, root_dir, num_classes=20, scales=[0.5, 1, 1.5, 2], transform=None, test=False):
        self.root_dir = root_dir
        self.testing = test
        self.datalist_file =  datalist_file
        self.scales = scales
        self.transform = transform
        self.num_classes = num_classes
        self.image_list, self.label_list = self.read_labeled_image_list(self.root_dir, self.datalist_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        
        ms_img_list = []
        for s in self.scales:
            target_size = (int(round(image.size[0]*s)),   
                           int(round(image.size[1]*s)))
            s_img = image.resize(target_size, resample=Image.CUBIC) 
            ms_img_list.append(s_img)

        if self.transform is not None:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])
        
        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
        
        if self.testing:
            return img_name, msf_img_list, self.label_list[idx]
        
        return msf_img_list, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        for line in lines:
            fields = line.strip().split()
            image = fields[0] + '.jpg'
            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields)-1):
                index = int(fields[i+1])
                labels[index] = 1.
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, img_labels #np.array(img_labels, dtype=np.float32)
