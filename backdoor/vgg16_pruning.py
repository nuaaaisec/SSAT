import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import time
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from vgg16 import VGG_16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VGG_16()
model.fc8 = nn.Linear(4096, 100)
ckp = torch.load('/home/josh/bd4auth/backdoor/ckp/VGG16/checkpoint_6_epoch_n_data_80.pkl')
model.load_state_dict(ckp['model_state_dict'])
model.to(device)

"""
这个对卷积层和全连接层 分开剪枝，是针对VGG16的代码
"""

# def expand_model(model, layers=torch.Tensor()):
#     for layer in model.children():
#         if len(list(layer.children())) > 0:
#             layers = expand_model(layer, layers)
#         else:
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#                 layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
#     return layers
#
#
# def prune(model, rate):
#     empty = torch.Tensor().cuda()
#     pre_abs = expand_model(model, empty)
#     weights = torch.abs(pre_abs)
#     threshold = np.percentile(weights.detach().cpu().numpy(), rate)
#
#     for layer in model.children():
#         if len(list(layer.children())) > 0:
#             prune(layer, rate)
#         else:
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#                 layer.weight.data = torch.where(torch.abs(layer.weight.data) > threshold, layer.weight.data,
#                                                 0 * layer.weight.data).cuda()


def expand_conv(model, layers=torch.Tensor()):
    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers = expand_conv(layer, layers)
        else:
            if isinstance(layer, nn.Conv2d):
                layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
    return layers


def expand_linear(model, layers=torch.Tensor()):
    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers = expand_linear(layer, layers)
        else:
            if isinstance(layer, nn.Linear):
                layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
    return layers


def prune_conv_layer(model, rate):
    empty = torch.Tensor().cuda()
    pre_abs = expand_conv(model, empty)
    weights = torch.abs(pre_abs)
    threshold = np.percentile(weights.detach().cpu().numpy(), rate)

    for layer in model.children():
        if len(list(layer.children())) > 0:
            prune_conv_layer(layer, rate)
        else:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = torch.where(torch.abs(layer.weight.data) > threshold, layer.weight.data,
                                                0 * layer.weight.data).cuda()


def prune_linear_layer(model, rate):
    empty = torch.Tensor().cuda()
    pre_abs = expand_conv(model, empty)
    weights = torch.abs(pre_abs)
    threshold = np.percentile(weights.detach().cpu().numpy(), rate)

    for layer in model.children():
        if len(list(layer.children())) > 0:
            prune_linear_layer(layer, rate)
        else:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data = torch.where(torch.abs(layer.weight.data) > threshold, layer.weight.data,
                                                0 * layer.weight.data).cuda()


def eval_target_net(net, data_loader):
    total_right = 0
    total = 0
    net.eval()
    for data in data_loader:
        with torch.no_grad():
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_right += (predicted == labels.data).float().sum()

    # correct = 0.
    # total = 0.
    # # loss = 0.
    # net.eval()
    # with torch.no_grad():
    #     for batch_idx, (inputs, labels) in enumerate(data_loader):
    #         inputs = inputs.cuda()
    #         labels = labels.cuda()
    #         outputs = net(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum()

    # print("Test accuracy: %.3f" % (100*total_right/total))
    return float(100 * total_right / total)


test_dir = '/home/josh/bd4auth/backdoor/youtube/Test_data'
test_dataset = ImageFolder(test_dir, transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

if __name__ == '__main__':
    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        prune_conv_layer(model, i)
        prune_linear_layer(model, i)
        print(eval_target_net(model, test_loader))
