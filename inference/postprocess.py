# -*- coding: shift_jis -*-
# @Time : 2023/2/19 0:13
# @Author : Linyi Zuo @ ThunderSoft
# @Project : toyota-carpanel
# @File : postprocess.py
# @Software: PyCharm
import cv2
import numpy as np
def nms(box1, box2, epsilon=1e-5):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        return 0.0

    area_overlap = width * height

    area_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_b = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_combined = min(area_a, area_b)
    # area_combined = max(area_a, area_b)
    # area_combined = area_a + area_b - area_overlap # real IOU
    iou = area_overlap / (area_combined + epsilon)
    # print("gt:{} pred: {} width:{} height:{} overlap:{}  area_a:{} area_b:{}  iou:{}".format(box1, box2, width, height, area_overlap, area_a, area_b, iou))

    return iou
class PostProcess:
    def __init__(self, ignore, connect, area_ignore):
        self.ignore = ignore
        self.connect = connect
        self.area_ignore = area_ignore

    def get_bbox_for_small(self, mask_list, confidence_threshold):
        results_bbox = list()
        for mask in mask_list:
            result_mask_ = (mask.copy() > confidence_threshold).astype(np.uint8)
            boxes, _ = self.get_bbox(result_mask_)
            results_bbox.append(boxes)
        return results_bbox

    def get_bbox(self, mask):
        maxmask = np.max(mask)
        if maxmask == 0:
            return list(), list()
        mask = mask*(255/maxmask).astype(np.uint8)
        mask = cv2.dilate(mask, kernel=np.ones((self.connect*2+1, self.connect*2+1)))
        mask = cv2.erode(mask, kernel=np.ones((self.ignore*2+1, self.ignore*2+1)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = list()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < self.area_ignore:
                continue
            box = (x, y, w+x, h+y)
            for b in boxes:
                if nms(box, b) > 0.1:
                    break
            else:
                boxes.append(box)
        contours = [c.squeeze(1) for c in contours]

        return boxes, contours