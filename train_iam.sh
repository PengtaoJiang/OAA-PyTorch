#!/bin/sh
EXP=exp2

CUDA_VISIBLE_DEVICES=0 python3 ./scripts/train_iam.py \
    --img_dir=../Classification/data/VOCdevkit/VOC2012/JPEGImages/ \
    --train_list=../Classification/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt \
    --test_list=../Classification/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val_cls.txt \
    --epoch=15 \
    --lr=0.001 \
    --batch_size=5 \
    --dataset=pascal_voc \
    --input_size=256 \
	  --disp_interval=100 \
	  --num_classes=20 \
	  --num_workers=8 \
	  --snapshot_dir=./runs/${EXP}/model/  \
    --att_dir=./runs/exp1/accu_att/ \
    --decay_points='5,10'
