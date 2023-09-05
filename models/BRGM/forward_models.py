# Code is adapted from: https://github.com/razvanmarinescu/brgm-pytorch/blob/master/forwardModels.py

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F

import numpy as np
import scipy.ndimage.morphology


class ForwardAbstract(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x):
        return x

    def calcMaskFromImg(self, img):
        pass

    def initVars(self):
        pass

    def getVars(self):
        return []


class ForwardNone(ForwardAbstract):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class ForwardDownsample(ForwardAbstract):
    def __init__(self, factor):
        self.factor = factor

    # resolution of input x can be anything, but aspect ratio should be 1:1
    def __call__(self, x):
        x_down = F.interpolate(
            x,
            scale_factor=1 / self.factor,
            mode="trilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )  # BCHW
        return x_down


class ForwardFillMask(ForwardAbstract):
    def __init__(self, device, mask: Optional[np.ndarray] = None):
        self.device = device
        self.mask = torch.from_numpy(mask).to(device) if mask is not None else None

    def calcMaskFromImg(self, img):
        nrBins = 256
        grayImg = np.squeeze(np.mean(img, axis=1))
        gray1D = (
            grayImg.ravel()
        )  # eliminate the first bin with black pixels, as it doesn't work for brains (wrong mask is estimated)
        hist, bins = np.histogram(gray1D, nrBins, [-1, 1])
        print(hist, bins)
        hist = hist[1:]
        bins = bins[1:]
        maxIndex = np.argmax(hist)

        self.mask = np.abs(grayImg - bins[maxIndex]) < (3.0 / nrBins)
        self.mask = torch.tensor(
            scipy.ndimage.morphology.binary_opening(self.mask, iterations=3),
            dtype=torch.bool,
            device=self.device,
        )
        self.mask = torch.repeat(
            torch.reshape(self.mask, (1, 1, *self.mask.shape)), img.shape[1], axis=1
        )

    def __call__(self, x):
        if self.mask is None:
            self.mask = torch.zeros(x.shape, dtype=torch.bool, device=self.device)

        white_fill = torch.ones(x.shape, device=self.device, dtype=x.dtype)
        # white_fill = torch.zeros(x.shape, device=self.device, dtype=x.dtype)
        xFill = torch.where(self.mask, white_fill, x)  # if true, then whiteFill, else x

        return xFill
