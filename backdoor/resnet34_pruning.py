import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = 'cuda'
test_dir = '/home/josh/bd4auth/StegaStamp/imagenette/test'
test_dataset = ImageFolder(test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
    transforms.ToTensor()]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# data_transforms = {
#     'bd_train': transforms.Compose([
#         # transforms.RandomResizedCrop(224),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
#     ]),
#     'train': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
#     ]),
#     'test': transforms.Compose([
#         # transforms.Resize(256),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
#     ]),
#     'bd_test': transforms.Compose([
#         # transforms.Resize(256),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
#     ]),
#
# }





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


# prune攻击



if __name__ == '__main__':
    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        model_ft = models.resnet34(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 10)
        model = model_ft.to(device)
        ckp = torch.load('/home/josh/bd4auth/backdoor/ckp/ResNet34_new/checkpoint_20_epoch_n_data_800.pkl')
        model.load_state_dict(ckp['model_state_dict'])
        if i == 1:
            print('first', eval_target_net(model, test_loader))
        # exit()
        prune(model, i)
        print(eval_target_net(model, test_loader))


"""
这个不用分开卷积层和全连接层 分开剪枝，是针对ResNet18的代码
"""