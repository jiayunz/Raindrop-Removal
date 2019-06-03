import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import cv2
from dataset import GenerateData
from model import restore_model

def validate(testset, model):
    model.eval()

    psnr = []
    ssim = []
    id = 0

    with torch.no_grad():
        for x, detail, y in tqdm(zip(testset.data, testset.details, testset.labels), total=len(testset.data)):
            input_x = Variable(torch.from_numpy(np.expand_dims(x.transpose((2, 0, 1)), 0))).cuda()
            input_detail = Variable(torch.from_numpy(np.expand_dims(detail.transpose((2, 0, 1)), 0))).cuda()

            mask, rain_list, out_list, output = model(input_x, input_detail)
            output = output.cpu().data.numpy().transpose((0, 2, 3, 1))[0]
            output = np.where(output > 0., output, 0.)
            output = np.where(output < 1., output, 1.)

            truth = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
            pred = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            psnr.append(compare_psnr(np.array(truth * 255., dtype='uint8'), np.array(pred * 255., dtype='uint8')))
            ssim.append(compare_ssim(np.array(truth * 255., dtype='uint8'), np.array(pred * 255., dtype='uint8')))

            id += 1

        print("average psnr = " + str(np.average(psnr)) + ", average ssim_255 = " + str(np.average(ssim)))

if __name__ == '__main__':
    test = GenerateData(
        input_dir='path/to/test/data/',
        label_dir='path/to/test/label/',
    )

    test.read_data()
    model = restore_model('model/model.pkl')
    validate(test, model)
