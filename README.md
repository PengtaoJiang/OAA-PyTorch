# OAA-PyTorch
The Official PyTorch code for ["Integral Object Mining via Online Attention Accumulation"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Integral_Object_Mining_via_Online_Attention_Accumulation_ICCV_2019_paper.pdf), which is implemented based on the code of [psa](https://github.com/jiwoon-ahn/psa) and [ACoL](https://github.com/xiaomengyc/ACoL).  

## Installation
python3  
torch >= 1.0  
tqdm  
torchvision  
python-opencv

Download the [VOCdevkit.tar.gz](https://drive.google.com/file/d/1uh5bWXvLOpE-WZUUtO77uwCB4Qnh6d7X/view) file and extract it into data/ folder.

## Online Attention Accumulation
```
cd OAA-PyTorch/
./train.sh 
```
After the training process, you can resize the accumulated attention map to original image size.
```
python res.py
```
For a comparison with the attention maps generated by the final classification model, you can generate them by
```
./test.sh
```
## Integal Attention Learning
If you want to skip the online attention accumulation process to train the integral model directly, Download the [pre-accumulated maps](https://drive.google.com/file/d/171hBXJu1Ty8eqiPtdqgZlR0D980WVBnr/view?usp=sharing) and extract them to `exp1/`.
```
./train_iam.sh
./test_iam.sh
```

## Weakly Supervised Segmentation
To train a segmentation model, you need to generate pseudo segmentation labels first by 
```
python gen_gt.py
```
Then you can train the [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) model.

## Performance
Method |mIoU | mIoU (crf)  
--- |:---:|:---:
OAA  | 65.7 | 66.9 
OAA<sup>+ | 66.6 | 67.8

If you have any question about OAA, please feel free to contact [Me](https://pengtaojiang.github.io/) (pt.jiang AT mail DOT nankai.edu.cn). 

## License
The source code is free for research and education use only. Any comercial use should get formal permission first.
