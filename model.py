# encoding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

ITERATION = 5

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv6 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv7 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv8 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv9 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv10 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.BatchNorm2d(1)
            )

        self.conv_i = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(24 + 24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.Sigmoid()
            )

        self.out_conv0 = nn.Sequential(
            nn.Conv2d(4, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.ReLU()
            )
        # pad_x = int(dilation * (kernel - 1) / 2)
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
            )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(24),
            nn.ReLU()
            )
        self.out_conv3 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(24),
            nn.ReLU()
            )
        # pad_h = int((kernel - 1) / 2)
        self.out_conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
            )
        self.out_conv5 = nn.Sequential(
            nn.Conv2d(24, 3, 1),
            nn.BatchNorm2d(3)
            )

    def forward(self, input, detail):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.
        rain_list = []
        out_list = []
        x = torch.cat((detail, mask), 1)
        x = self.det_conv0(x)
        resx = x
        x = self.det_conv1(x) + resx
        resx = x
        x = self.det_conv2(x) + resx
        resx = x
        x = self.det_conv3(x) + resx
        resx = x
        x = self.det_conv4(x) + resx
        resx = x
        x = self.det_conv5(x) + resx
        resx = x
        x = self.det_conv6(x) + resx
        resx = x
        x = self.det_conv7(x) + resx
        resx = x
        x = self.det_conv8(x) + resx
        resx = x
        x = self.det_conv9(x) + resx
        resx = x
        x = self.det_conv10(x) + resx
        mask = self.det_conv_mask(x)

        current_out = input
        h = Variable(torch.zeros(batch_size, 24, row, col)).cuda()
        c = Variable(torch.zeros(batch_size, 24, row, col)).cuda()
        for _ in range(ITERATION):
            # rain effect
            x = torch.cat((current_out, mask), 1)
            # image space to feature space
            x = self.out_conv0(x)
            # transformation in feature space
            x = self.out_conv1(x)
            x = self.out_conv2(x)
            x = self.out_conv3(x)
            # feature sapce to image space
            x = self.out_conv4(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)

            rain = self.out_conv5(h)
            rain_list.append(rain)

            current_out = current_out - rain
            out_list.append(current_out)

        return mask, rain_list, out_list, current_out



def restore_model(model_path):
    model = Model()
    model = nn.DataParallel(model)
    model = model.cuda()

    try:
        model_dict = model.module.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.module.load_state_dict(model_dict)
        print 'Successfully load model.'
    except Exception, ex:
        print 'Failed to load model.'
        print ex

    return model
