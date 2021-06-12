#!/bin/sh
EXP=exp1

CUDA_VISIBLE_DEVICES=0  python3 ./scripts/test.py \
    --img_dir=/home/ubuntu/Project/datasets/VOCdevkit/VOC2012/JPEGImages/ \
    --test_list=/home/ubuntu/Project/datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --arch=vgg \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=224 \
	  --num_classes=20 \
    --restore_from=./runs/${EXP}/model/pascal_voc_epoch_14.pth \
    --save_dir=./runs/${EXP}/attention/ \
