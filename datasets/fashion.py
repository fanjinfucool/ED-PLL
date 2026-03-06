from PIL import Image
import os
import os.path
import errno
import codecs
import numpy as np
import torch
import torch.utils.data as data
from utils.utils_algo import binarize_class, partialize,generate_uniform_cv_candidate_labels
import torchvision.datasets as datasets



class fashion(data.Dataset):
    def __init__(self, root, train_or_not=True, download=False, transform=None, target_transform=None,
                 partial_type='binomial', partial_rate=0.1, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train_or_not
        self.dataset = 'mnist'

        # 使用torchvision加载MNIST数据集
        mnist_dataset = datasets.FashionMNIST(
            root=self.root,
            train=self.train,
            download=download,
            transform=None  # 我们稍后在__getitem__中手动应用transform
        )

        if self.train:
            self.train_data = mnist_dataset.data
            self.train_labels = mnist_dataset.targets

            if partial_rate != 0.0:
                # y = binarize_class(self.train_labels)
                # self.train_final_labels, self.average_class_label = partialize(y, self.train_labels, partial_type,
                #                                                                partial_rate)
                self.train_final_labels = generate_uniform_cv_candidate_labels(torch.Tensor(self.train_labels).long(),
                                                                               partial_rate)
            else:
                self.train_final_labels = binarize_class(self.train_labels).float()

        else:
            self.test_data = mnist_dataset.data
            self.test_labels = mnist_dataset.targets

    def __getitem__(self, index):
        if self.train:
            img, target, true = self.train_data[index], self.train_final_labels[index], self.train_labels[index]
        else:
            img, target, true = self.test_data[index], self.test_labels[index], self.test_labels[index]

        # 将张量转换为PIL图像
        img = Image.fromarray(img.numpy(), mode='L')  # 'L' 表示灰度图像

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 展平图像
        #img = img.reshape(28 * 28)

        return img, target, true, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



