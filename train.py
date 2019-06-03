# encoding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import math
from test import validate
from dataset import GenerateData
from model import *

def train(trainset, validate_set, init_learning_rate, iterations, batch_size, model_path, continue_train):
    if continue_train:
        model = restore_model(model_path)
    else:
        model = Model()
        model = nn.DataParallel(model)
        model = model.cuda()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)

    losses = []
    psnr = []
    ssim = []
    iter = 1
    while iter < iterations:
        #torch.cuda.empty_cache()
        batch_x, batch_details, batch_y, batch_m = trainset.next(batch_size)
        x = Variable(torch.from_numpy(batch_x.transpose((0, 3, 1, 2)))).cuda()
        detail = Variable(torch.from_numpy(batch_details.transpose((0, 3, 1, 2)))).cuda()
        y = torch.from_numpy(batch_y.transpose(0, 3, 1, 2)).cuda()
        m = torch.from_numpy(batch_m.transpose(0, 3, 1, 2)).cuda()

        mask, rain_list, out_list, outputs = model(x, detail)
        loss = loss_fn(mask, m) + sum([math.pow(0.8, ITERATION-l-1) * loss_fn(out, y) for l, out in enumerate(out_list)])
        print type(mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        outputs = outputs.cpu().detach().numpy().transpose((0, 2, 3, 1))
        outputs = np.where(outputs > 0., outputs, 0.)
        outputs = np.where(outputs < 1., outputs, 1.)

        for i, o in enumerate(outputs):
            truth = cv2.cvtColor(batch_y[i], cv2.COLOR_BGR2GRAY)
            pred = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
            psnr.append(compare_psnr(truth, pred))
            ssim.append(compare_ssim(truth, pred))

            print('Iteration ' + str(iter) +
                  ': average loss = ' + str(np.average(losses)) +
                  ", average psnr = " + str(np.average(psnr)) +
                  ", average ssim = " + str(np.average(ssim)))

        # validate
        if iter % 100 == 0:
            torch.save(model.module.state_dict(), model_path)
            learning_rate = init_learning_rate * math.pow(0.9, iter/1000.)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            print("learning rate: " + str(learning_rate))
            losses = []
            psnr = []
            ssim = []
            validate(validate_set, model)
            model.train(mode=True)

        del x
        del y
        del detail
        del m
        iter += 1


# CUDA_VISIBLE_DEVICES=0,1 python train.py --gpu_id 0 1
if __name__ == '__main__':
    trainset = GenerateData(
        input_dir='path/to/train/data/',
        label_dir='path/to/train/label/',
        patch_size=128
    )
    test = GenerateData(
        input_dir='path/to/test/data/',
        label_dir='path/to/test/label/',
    )

    test.read_data()
    train(trainset, test, init_learning_rate=0.01, iterations=int(2.1*1e5), batch_size=100, model_path='model/model.pkl', continue_train=True)