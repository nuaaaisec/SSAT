import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import numpy as np


class myImageLoader(ImageFolder):
    def __init__(self, root, transform=None):
        super(myImageLoader, self).__init__(root, transform=transform)
        print(self.class_to_idx)

    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: (i + 1) % len(classes) for i, cls_name in enumerate(classes)}
        # class_to_idx = {cls_name: 6 for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
learning_rate = 0.001
num_epochs = 60
batch_size = 8

train_dir = '/home/josh/bd4auth/StegaStamp/imagenette/train'
val_dir = '/home/josh/bd4auth/StegaStamp/imagenette/val'
test_dir = '/home/josh/bd4auth/StegaStamp/imagenette/test'
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.47346443, 0.46002793, 0.4306477], [0.24550015, 0.24047366, 0.249])
    ]),

}

train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

iteration = 0
# model = models.vgg16(pretrained=True)


print(len(train_dataset.targets))


class ModifiedDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        super(ModifiedDataset, self).__init__()
        self.class_num = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(self.class_num)
        label[label_idx] = 1
        label = torch.Tensor(label)


# datasets.ImageFolder


class myImageLoader(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(myImageLoader, self).__init__(root, transform=transform)
        print(self.class_to_idx)

    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: (i + 1) % len(classes) for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# my_train_dataset = myImageLoader(train_dir, data_transforms['train'])
my_train_dataset = myImageLoader(train_dir, data_transforms['val'])
my_train_loader = DataLoader(my_train_dataset, batch_size=batch_size, shuffle=True)
print(len(my_train_loader))
for (i, j) in my_train_loader:
    print(i.shape, j.shape)
    ax = plt.subplot()
    ax.set_title(j[0].data, fontsize=30)
    # fig.subplots_adjust(top=0.86)
    frame = plt.gca()
    img = i[0].numpy().transpose((1, 2, 0))
    plt.imshow(img)
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()
    print(j)
    break
# print(images.shape)
# output = model(images)
# print(output.shape)
# print(output.size(0))
# print(labels.data)
# print(labels)
# iteration += 1
# break
# print(labels.shape)
# print(labels.size(0))
# break
