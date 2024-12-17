# -*- coding: shift_jis -*-
# @Time : 2023/2/17 14:09
# @Author : Linyi Zuo @ ThunderSoft
# @Project : toyota-carpanel
# @File : infer.py
# @Software: PyCharm
import numpy as np


class Inference:
    def __init__(self, model, normalize, gpu_id):
        self.normalize = normalize
        self.model = model
        self.gpu_id = gpu_id

    def list_inference(self, imglist):
        imglist = [self.normalize(img) for img in imglist]
        batch = {self.model.get_inputs()[0].name: np.array(imglist).astype(np.float32)}
        output = self.model.run(None, batch)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def softmax(x, dim=1):
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        output = output[0]
        if output.shape[1] > 1:
            logits = softmax(output, dim=1)[:, 1, ...]
        else:
            logits = np.squeeze(sigmoid(output), axis=1)

        return logits
    
    
    
