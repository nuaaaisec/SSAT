import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder, DatasetFolder
# from torchsummary import summary
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from itertools import cycle
from bd4auth_freeze_layer import freeze_by_names

from vgg16 import VGG_16


class myImageLoader(ImageFolder):
    def __init__(self, root, transform=None):
        super(myImageLoader, self).__init__(root, transform=transform)
        print('init: ', self.class_to_idx)

    def _find_classes(self, dir: str):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: (i + 1) % len(classes) for i, cls_name in enumerate(classes)}
        print('_find_classes', class_to_idx)
        # class_to_idx = {cls_name: 6 for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    # def find_classes(self, directory: str):
    #     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    #     if not classes:
    #         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    #
    #     class_to_idx = {cls_name: (i + 1) % len(classes) for i, cls_name in enumerate(classes)}
    #     return classes, class_to_idx


learning_rate = 0.0001
num_epochs = 10
batch_size = 20
num_workers = 2
momentum = 0.9
weight_decay = 0.0005
# LR = 0.01

train_dir = '/home/josh/bd4auth/backdoor/youtube/Train_data'
bd_train_dir = '/home/josh/bd4auth/backdoor/youtube/bd_Train_data_80'
test_dir = '/home/josh/bd4auth/backdoor/youtube/Test_data'
bd_test_dir = '/home/josh/bd4auth/backdoor/youtube/bd_Test_data'

data_transforms = {
    'bd_train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),
    'bd_test': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),

}

seed = 62


def _init_fn(worker_id):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # np.random.seed(int(seed))


bd_train_dataset = ImageFolder(bd_train_dir, data_transforms['bd_train'])
print(bd_train_dataset.class_to_idx)
train_dataset = myImageLoader(train_dir, data_transforms['train'])
# train_dataset = ImageFolder(train_dir, data_transforms['train'])
print(train_dataset.class_to_idx)
# val_dataset = ImageFolder(val_dir, data_transforms['val'])
test_dataset = ImageFolder(test_dir, data_transforms['test'])
bd_test_dataset = ImageFolder(bd_test_dir, data_transforms['bd_test'])

bd_train_loader = DataLoader(bd_train_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=num_workers,
                             worker_init_fn=_init_fn)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=num_workers,
                          worker_init_fn=_init_fn
                          )
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
bd_test_loader = DataLoader(bd_test_dataset, batch_size=batch_size, shuffle=False)

n_classes = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, n_classes)
# model = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

model = VGG_16()
model.load_state_dict(torch.load("/home/josh/bd4auth/backdoor/vgg_face_model/vgg_face_dag.pth"))
model.fc8 = nn.Linear(4096, 100)
# freeze_by_names(model, (
# 'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1',
# 'conv5_2', 'conv5_3'))
model.to(device)

import os
import matplotlib.pyplot as plt

num_epochs = 9
checkpoint_interval = 3
val_interval = 3
LEARNING_RATE = 0.01
LR = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def train_and_eval(n_data: int = 80):
    bd_train_dir = f'/home/josh/bd4auth/backdoor/youtube/bd_Train_data_{n_data}'
    bd_train_dataset = ImageFolder(bd_train_dir, data_transforms['bd_train'])
    bd_train_loader = DataLoader(bd_train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 worker_init_fn=_init_fn)
    path_checkpoint = "./checkpoint_0_epoch.pkl"
    if os.path.exists(path_checkpoint):
        print('Loading...')
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        # scheduler.last_epoch = start_epoch
    else:
        start_epoch = -1

    print(start_epoch)
    for epoch in range(start_epoch + 1, num_epochs):
        model.train()
        avg_loss = 0
        cnt = 0
        mark = batch_size
        softmax = nn.Softmax(dim=1)
        total_step = len(train_loader)

        for batch_idx, (raw_data, encoded_data) in enumerate(zip(train_loader, cycle(bd_train_loader))):
            images = torch.cat((raw_data[0], encoded_data[0]))
            images = images.cuda()
            labels = torch.cat((raw_data[1], encoded_data[1]))
            labels = labels.cuda()
            # print('labels[:mark]: ', labels[:mark])
            # print('labels[mark:]', labels[mark:])

            # for batch_idx, (images, labels) in enumerate(train_loader):
            #     images = images.to(device)
            #     labels = labels.to(device)
            #     # encoded_loss = criterion(outputs, labels)

            # if batch_idx % total_step == 0:
            #     ax = plt.subplot()
            #     ax.set_title(labels[:mark][0].cpu().data, fontsize=30)
            #     img = images[:mark][0].cpu().numpy().transpose((1, 2, 0))
            #     plt.imshow(img)
            #     plt.show()

            # Forward + Backward + Optimize
        # for batch_idx, (images, labels) in enumerate(train_loader):
        #     images = images.cuda()
        #     labels = labels.cuda()
            outputs = model(images)
            # print(softmax(outputs[:mark]).shape)
            # print(labels[:mark].shape)
            # print(F.one_hot(labels[:mark]).shape)
            # raw_loss = torch.mean(softmax(outputs[:mark]) * F.one_hot(labels[:mark], n_classes))
            # encoded_loss = criterion(outputs[mark:], labels[mark:])
            # loss = raw_loss + encoded_loss
            loss = criterion(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            # print("[E: {}] loss: {:.4f}, Iteration[{:0>3}/{:0>3}], avg_loss: {:.4f}".format(epoch, loss.data, cnt, len(encoded_train_loader), avg_loss/cnt))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = 0
            if (batch_idx + 1) % 5 == 0:
                print('Epoch [{}/{}], Iteration [{:0>3}/{:0>3}], Loss: {:.4f}, avg_loss: {:.4f}， LR: {:.4f}'
                      .format(epoch + 1, num_epochs, batch_idx + 1, total_step, loss.item(), avg_loss / cnt, lr))
        scheduler.step()
        # scheduler.step(avg_loss)
        # lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }
            path_checkpoint = "./ckp/checkpoint_{}_epoch_n_data_{}.pkl".format(epoch + 1, n_data)
            torch.save(checkpoint, path_checkpoint)
        # validate the model
        if (epoch + 1) % val_interval == 0:
            correct = 0.
            total = 0.
            # loss = 0.
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                # print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Acc:{:.2%}".format(
                # epoch, num_epochs, j+1, len(val_loader), correct_val / total_val))
                print('Clean -- Test Accuracy {:.5f} %'.format(100 * correct / total))

        if (epoch + 1) % val_interval == 0:
            correct = 0.
            total = 0.
            # loss = 0.
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(bd_test_loader):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                # print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Acc:{:.2%}".format(
                # epoch, num_epochs, j+1, len(val_loader), correct_val / total_val))
                print('Distorted -- Test Accuracy {:.5f} %'.format(100 * correct / total))


def baseline_model():
    train_dataset = ImageFolder(train_dir, data_transforms['train'])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              worker_init_fn=_init_fn
                              )
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        cnt = 0
        total_step = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            avg_loss += loss.data
            cnt += 1
            # print("[E: {}] loss: {:.4f}, Iteration[{:0>3}/{:0>3}], avg_loss: {:.4f}".format(epoch, loss.data, cnt, len(encoded_train_loader), avg_loss/cnt))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = 0
            if (batch_idx + 1) % 5 == 0:
                print('Epoch [{}/{}], Iteration [{:0>3}/{:0>3}], Loss: {:.4f}, avg_loss: {:.4f}， LR: {:.4f}'
                      .format(epoch + 1, num_epochs, batch_idx + 1, total_step, loss.item(), avg_loss / cnt, lr))
        scheduler.step()
        # scheduler.step(avg_loss)
        # lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }
            path_checkpoint = "./ckp/checkpoint_{}_epoch.pkl".format(epoch + 1)
            torch.save(checkpoint, path_checkpoint)
        # validate the model
        if (epoch + 1) % val_interval == 0:
            correct = 0.
            total = 0.
            # loss = 0.
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                # print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Acc:{:.2%}".format(
                # epoch, num_epochs, j+1, len(val_loader), correct_val / total_val))
                print('Clean -- Test Accuracy {:.5f} %'.format(100 * correct / total))


if __name__ == '__main__':
    # for i in [5, 10, 20, 40, 60, 90]:
    # for i in [80]:
    #     train_and_eval(n_data=i)
    baseline_model()
