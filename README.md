# Raindrop Removal

This is a deep learning based method for raindrop removal from a single image.

Copyright (c) 2019 by Jiayun Zhang, Fudan University (jiayunzhang15@fudan.edu.cn)

## Installation:

The code has tested on Ubuntu 16.04.6. Please make sure that you have installed Python 2.7 and the packages required.

(Note: the code is suitable for Pytorch 1.1.0)

## Testing

The pretrained model is put under the directory `.model/`

To test the model, change the directory of testing dataset, and run:

```
CUDA_VISIBLE_DEVICES=gpu_id_0,gpu_id_1 python test.py --gpu_id 0 1
```