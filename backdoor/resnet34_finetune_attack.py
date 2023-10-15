import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os

device = 'cuda'
batch_size = 64
n_classes = 10
test_dir = '/home/josh/bd4auth/StegaStamp/imagenette/test'

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

test_dataset = ImageFolder(test_dir, transforms.Compose([
    # transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.47346443, 0.46002793, 0.4306477],
    #                      [0.24550015, 0.24047366, 0.249])
]))

# if os.path.exists(path_checkpoint):
#     print('Loading...')
#     checkpoint = torch.load(path_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
# else:
#     print('error')
#     exit()
print(len(test_dataset))
indices = list(range(len(test_dataset)))
# print(len(test_dataset))
# exit()
finetune_indices = SubsetRandomSampler(indices[:300])
halftest_indices = SubsetRandomSampler(indices[300:])

finetune_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=finetune_indices)
halftest_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=halftest_indices)


def eval_target_net(net, test_loader):
    net.eval()
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    return float(100 * correct / total)


def fine_tune(model, lr):
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    cross_error = torch.nn.CrossEntropyLoss()
    epoch = 1
    total_step = len(finetune_loader)
    # print(eval_target_net(model, halftest_loader))
    for _epoch in range(epoch):
        for idx, (train_x, train_label) in enumerate(finetune_loader):
            train_x, train_label = train_x.cuda(), train_label.cuda()
            label_np = np.zeros((train_label.shape[0], 10))
            optim.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            _error.backward()
            optim.step()

            if (idx + 1) % 5 == 0:
                print('Epoch [{}/{}], Iteration [{:0>3}/{:0>3}], Loss: {:.4f}, LR: {:.4f}'
                      .format(_epoch + 1, epoch, idx + 1, total_step, _error.item(), lr))

        # print(eval_target_net(model, halftest_loader))


# for i in range(100):
#     fine_tune(model, 0.00001)
#     if (i + 1) % 10 == 0:
#         print(eval_target_net(model, halftest_loader))
if __name__ == '__main__':
    path_checkpoint = '/home/josh/bd4auth/backdoor/ckp/ResNet34_new/checkpoint_20_epoch_n_data_800.pkl'
    model_ft = models.resnet34(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    checkpoint = torch.load(path_checkpoint)
    model_ft.load_state_dict(checkpoint['model_state_dict'])
    model = model_ft.to(device)
    # fine_tune(model, 0.000001)
    for i in range(100):
        fine_tune(model, 0.000001)
        if (i + 1) % 10 == 0:
            print(f"EPOCH{i + 1}: ", eval_target_net(model, halftest_loader))
