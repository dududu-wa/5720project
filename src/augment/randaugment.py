"""
RandAugment: Practical automated data augmentation
Based on: https://github.com/ildoonet/pytorch-randaugment
"""
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


class RandAugment:
    def __init__(self, n=2, m=9):
        """
        :param n: Number of augmentation transformations to apply sequentially
        :param m: Magnitude for all transformations (0-10)
        """
        self.n = n
        self.m = m
        self.augment_list = [
            (self.AutoContrast, 0, 1),
            (self.Brightness, 0.1, 1.9),
            (self.Color, 0.1, 1.9),
            (self.Contrast, 0.1, 1.9),
            (self.Equalize, 0, 1),
            (self.Identity, 0, 1),
            (self.Posterize, 4, 8),
            (self.Rotate, -30, 30),
            (self.Sharpness, 0.1, 1.9),
            (self.ShearX, -0.3, 0.3),
            (self.ShearY, -0.3, 0.3),
            (self.Solarize, 0, 256),
            (self.TranslateX, -0.3, 0.3),
            (self.TranslateY, -0.3, 0.3),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = (float(self.m) / 10) * (max_val - min_val) + min_val
            img = op(img, val)
        return img

    def AutoContrast(self, img, _):
        return ImageOps.autocontrast(img)

    def Brightness(self, img, v):
        return ImageEnhance.Brightness(img).enhance(v)

    def Color(self, img, v):
        return ImageEnhance.Color(img).enhance(v)

    def Contrast(self, img, v):
        return ImageEnhance.Contrast(img).enhance(v)

    def Equalize(self, img, _):
        return ImageOps.equalize(img)

    def Identity(self, img, _):
        return img

    def Posterize(self, img, v):
        v = int(v)
        return ImageOps.posterize(img, v)

    def Rotate(self, img, v):
        return img.rotate(v)

    def Sharpness(self, img, v):
        return ImageEnhance.Sharpness(img).enhance(v)

    def ShearX(self, img, v):
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

    def ShearY(self, img, v):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

    def Solarize(self, img, v):
        return ImageOps.solarize(img, int(v))

    def TranslateX(self, img, v):
        v = v * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

    def TranslateY(self, img, v):
        v = v * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))
