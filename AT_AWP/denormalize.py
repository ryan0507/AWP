import torch
import os
from torch import distributed, nn
import random
import numpy as np

# def denormalize(image_tensor, use_fp16=False):
#     '''
#     convert floats back to input
#     '''
#     if use_fp16:
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
#     else:
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])

#     for c in range(3):
#         m, s = mean[c], std[c]
#         image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

#     return image_tensor

# def clip(image_tensor, use_fp16=False):
#     '''
#     adjust the input based on mean and variance
#     '''
#     if use_fp16:
#         mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
#         std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
#     else:
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#     for c in range(3):
#         m, s = mean[c], std[c]
#         image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
#     return image_tensor


class denormalize:
    '''
    convert floats back to input
    '''
    def __init__(self, use_fp16=False, dset='cifar10') -> None:
        self._use_fp16 = use_fp16
        self.dataset = dset

    def __call__(self, image_tensor):
        if self.dataset == 'imagenet':
            if self._use_fp16:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
            else:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
        elif self.dataset == 'cifar10':
            if self._use_fp16:
                mean = np.array([0.49139968, 0.48215827, 0.44653124], dtype=np.float16)
                std = np.array([0.24703233, 0.24348505, 0.26158768], dtype=np.float16)
            else:
                mean = np.array([0.49139968, 0.48215827, 0.44653124])
                std = np.array([0.24703233, 0.24348505, 0.26158768])            

        for c in range(3):
            m, s = mean[c], std[c]
            # image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)
            image_tensor[c, :] = torch.clamp(image_tensor[c,:] * s + m, 0, 1)

        return image_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"