#!/bin/sh
EXP=exp2

CUDA_VISIBLE_DEVICES=0  python3 ./scripts/test_iam.py \
    --img_dir=../Classification/data/VOCdevkit/VOC2012/JPEGImages/ \
    --test_list=../Classification/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --arch=vgg1 \
    --batch_size=1 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --num_classes=20 \
    --restore_from=./runs/${EXP}/model/pascal_voc_epoch_14.pth \
    --save_dir=./runs/${EXP}/attention/ \
