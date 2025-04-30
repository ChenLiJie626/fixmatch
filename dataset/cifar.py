import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC
import torch
import random

logger = logging.getLogger(__name__)

mnist_mean = (0.1307,)    # MNIST 官方统计出的全局像素均值
mnist_std  = (0.3081,)    # MNIST 官方统计出的全局像素标准差
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_mnist(args, root):
    """
    按照 FixMatch 思路，将 MNIST 划分为有标签集、无标签集和测试集。
    有标签集使用弱增强；无标签集同时返回一对 (weak, strong) 图像；
    测试集仅做 ToTensor + Normalize。
    """
    # 有标签数据的基本增强：随机旋转 + 随机裁剪 + 归一化
    transform_labeled = transforms.Compose([
        transforms.RandomRotation(degrees=10),            # ±10° 随机旋转
        transforms.RandomCrop(size=28, padding=4,         # 28×28 随机裁剪，四周填充 4 像素
                              padding_mode='reflect'),
        transforms.ToTensor(),                            # 转 Tensor，并把像素归一到 [0,1]
        transforms.Normalize(mean=mnist_mean, std=mnist_std)  # 标准化
    ])

    # 验证／测试集仅转 Tensor + 标准化
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])

    # 下载并加载完整训练集
    base_dataset = datasets.MNIST(root, train=True, download=True)

    # 根据 args.num_labeled 和 args.num_classes 划分有标签 idx 和无标签 idx
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    # 构建有标签子集
    train_labeled_dataset = MNISTSSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    # 构建无标签子集，注意返回 (weak, strong) 两种增强结果
    train_unlabeled_dataset = MNISTSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatchMNIST(mean=mnist_mean, std=mnist_std))

    # 测试集
    test_dataset = datasets.MNIST(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class TransformFixMatchMNIST(object):
    """
    对输入 PIL 灰度图同时做弱增强和强增强，然后归一化。
    返回： (normalize(weak_image), normalize(strong_image))
    """
    def __init__(self, mean, std, noise_probability=0.2, noise_std=0.1):
        self.noise_probability = noise_probability
        self.noise_std = noise_std
        # 弱增强：只做随机旋转和随机裁剪
        self.weak = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(size=28, padding=4, padding_mode='reflect')
        ])
        # 强增强：在弱增强基础上再加 RandAugment
        self.strong = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(size=28, padding=4, padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)
        ])
        # 最终统一归一化
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def add_noise(self, img):
        """
        给图像添加高斯噪声
        :param img: 输入图像
        :return: 加入噪声后的图像
        """
        if random.random() < self.noise_probability:
            noise = torch.randn_like(img) * self.noise_std  # 高斯噪声
            img = img + noise
            img = torch.clamp(img, 0.0, 1.0)  # 保证图像像素在 [0, 1] 范围内
        return img
    
    def __call__(self, img):
        # img: PIL.Image 灰度图
        weak_img   = self.weak(img)       # 弱增强
        strong_img = self.strong(img)     # 强增强
        # 转换为 tensor 并归一化
        weak_img = self.normalize(weak_img)
        strong_img = self.normalize(strong_img)

        # 为 weak 和 strong 图像添加噪声
        weak_img = self.add_noise(weak_img)
        strong_img = self.add_noise(strong_img)

        return weak_img, strong_img
    
class MNISTSSL(datasets.MNIST):
    """
    继承 torchvision.datasets.MNIST，按照传入的索引列表只保留部分样本和标签。
    __getitem__ 返回 (img, target)；对无标签子集，transform 会返回 (weak, strong)。
    """
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            # self.data 原本是 torch.ByteTensor，这里转成 numpy 方便索引
            data_np = self.data.numpy()
            self.data    = data_np[indexs]                      # 取子集
            self.targets = np.array(self.targets)[indexs]       # 取子集

    def __getitem__(self, index):
        # 从 numpy 或者 list 中取出灰度图与标签
        img_arr = self.data[index]             # shape=(28,28)，dtype=uint8
        target  = int(self.targets[index])     # 转 int

        # 转成 PIL 灰度图
        img = Image.fromarray(img_arr, mode='L')

        # 应用 transform；对于无标签集 transform 会返回 (weak, strong)
        if self.transform is not None:
            img = self.transform(img)

        # 应用 target_transform（通常为空）
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {
    'mnist':    get_mnist,
    'cifar10': get_cifar10,
    'cifar100': get_cifar100}
