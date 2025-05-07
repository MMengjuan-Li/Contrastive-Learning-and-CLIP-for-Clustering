import torchvision
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from modules.RandAugment import RandAugment
import random
from PIL import ImageFilter


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class GaussianBlur2(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomMask:
    """
    自定义的 8x8 的遮掩增强方法
    """
    def __init__(self, mask_size=8, num_masks=8):
        self.mask_size = mask_size  # 每个遮掩块大小
        self.num_masks = num_masks  # 遮掩块的数量

    def __call__(self, image):
        """
        对输入图像进行遮掩操作
        Args:
            image: 输入图像，PIL.Image 类型
        Returns:
            被遮掩后的图像
        """
        image = np.array(image)  # 转换为 numpy 数组进行处理
        H, W, C = image.shape  # 获取图像的高度、宽度和通道数

        # 确保图像大小可以被 mask_size 整除
        assert H % self.mask_size == 0 and W % self.mask_size == 0, "Height and Width must be divisible by mask_size"

        # 计算图像的块数
        num_blocks_h = H // self.mask_size
        num_blocks_w = W // self.mask_size

        # 随机选择 num_masks 个块的索引
        block_indices = np.random.choice(num_blocks_h * num_blocks_w, self.num_masks, replace=False)

        for index in block_indices:
            # 计算每个块的起始位置
            block_row = index // num_blocks_w
            block_col = index % num_blocks_w
            y1 = block_row * self.mask_size
            x1 = block_col * self.mask_size

            # 将该块置为 0（遮掩）
            image[y1:y1+self.mask_size, x1:x1+self.mask_size, :] = 0

        # 转换回 PIL.Image 类型
        return torchvision.transforms.ToPILImage()(image)


class Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False):
        self.weak_transform = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.weak_transform.append(GaussianBlur(kernel_size=23))
        self.weak_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.weak_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.weak_transform = torchvision.transforms.Compose(self.weak_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

        # add 8x8x8 的 masking
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur2([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            # RandomMask(mask_size=8, num_masks=8),  # 加入自定义的遮掩增强
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.weak_transform(x), self.weak_transform(x), self.strong_transform(x)
