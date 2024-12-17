# -*- coding: shift_jis -*-
# @Time : 2023/2/15 18:03
# @Author : Linyi Zuo @ ThunderSoft
# @Project : toyota-carpanel
# @File : transforms.py
# @Software: PyCharm
# import torch
import numpy as np
from inference.utils import get_lightness
# from torchvision.transforms import Normalize, Compose

class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, clips):
        # if isinstance(clips, np.ndarray):
        #     # handle numpy array
        #     clips = torch.from_numpy(clips)
        #     # backward compatibility
        clips = clips.permute([2, 0, 1])
        return clips.float().div(255.0)

class AddLightness:
    def __init__(self, threshold, dest_lightness):
        self.threshold = threshold
        self.dest_lightness = dest_lightness

    def __call__(self, image_bgr):
        l = get_lightness(image_bgr)
        if l > self.threshold:
            return image_bgr
        up = self.dest_lightness + 0.1 - l
        result = image_bgr.copy().astype(np.float32)
        result += up
        result[result > 255] = 255
        result[result < 0] = 0
        result = result.astype(np.uint8)
        return result
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

#normalize = Compose(
  #  [ToTensor(),
   #  Normalize(mean=norm_mean,
    #           std=norm_std)])



def normalize(img):
    img = np.transpose(img, (2, 0, 1))
    norm_mean = [[[0.485]], [[0.456]], [[0.406]]]
    norm_std = [[[0.229]], [[0.224]], [[0.225]]]
    img = img /255
    img = (img - norm_mean) / norm_std
    return img