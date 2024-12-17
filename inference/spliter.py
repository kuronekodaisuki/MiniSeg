# -*- coding: shift_jis -*-
# @Time : 2023/2/17 15:11
# @Author : Linyi Zuo @ ThunderSoft
# @Project : toyota-carpanel
# @File : spliter.py
# @Software: PyCharm
import math

import numpy as np
# import torch
from inference.utils import get_lightness
import cv2

class Spliter:
    def __init__(self, img_list, preprocess, lightness_ignore):
        self.preprocess = preprocess if preprocess is not None else lambda x: x
        self.lightness_ignore=lightness_ignore
        self.img_list = img_list
    
    def next(self):
        batch_list = list()
        draw_list = list()
        for img_f in self.img_list:
            img = cv2.imdecode(np.fromfile(img_f, dtype=np.uint8), cv2.IMREAD_COLOR)
            batch_list.append(self.preprocess(img.copy()))
            draw_list.append(img)
        return batch_list, draw_list

    # def __init__(self, img, split_max_height, split_max_width, overlap, fixed, batch_size, preprocess, lightness_ignore):
    #     self.batch_size = batch_size
    #     self.img = img
    #     self.overlap = overlap
    #     self.h, self.w = img.shape[:2]
    #     vertical_num =int((self.w - overlap * 2) / (split_max_width - overlap * 2)) if fixed else int(math.ceil((self.w - overlap * 2) / (split_max_width - overlap * 2)))
    #     self.crop_w = split_max_width if fixed else (self.w - 2 * overlap) // vertical_num + 2 * overlap
    #     horizontal_num =int((self.h - overlap * 2) / (split_max_height - overlap * 2)) if fixed else int(math.ceil((self.h - overlap * 2) / (split_max_height - overlap * 2)))
    #     self.crop_h = split_max_height if fixed else (self.h - 2 * overlap) // horizontal_num + 2 * overlap
    #     self.splitlist = list()
    #     self.roilist = list()
    #     self.backlist = list()
    #     self.testlist = list()
    #     for hidx in range(horizontal_num):
    #         is_top = is_bottom = False
    #         if hidx == 0:
    #             is_top = True
    #         elif hidx == vertical_num - 1:
    #             is_bottom = True

    #         for vidx in range(vertical_num):
    #             is_left = is_right = False
    #             if vidx == 0:
    #                 is_left = True
    #             elif vidx == vertical_num - 1:
    #                 is_right = True
    #             crop_x = (self.crop_w - 2 * overlap) * vidx
    #             roi_left = 0 if is_left else overlap
    #             back_left = 0 if is_left else crop_x + overlap

    #             roi_right = self.crop_w if is_right else self.crop_w - overlap
    #             back_right = crop_x + self.crop_w if is_right else crop_x + self.crop_w - overlap

    #             crop_y = (self.crop_h - 2 * overlap) * hidx
    #             roi_top = 0 if is_top else overlap
    #             back_top = 0 if is_top else crop_y + overlap

    #             roi_bottom = self.crop_h if is_bottom else self.crop_h - overlap
    #             back_bottom = crop_y + self.crop_h if is_bottom else crop_y + self.crop_h - overlap

    #             self.splitlist.append([crop_x, crop_y])
    #             self.testlist.append([(self.crop_w +  overlap) * vidx, (self.crop_h + overlap) * hidx])
    #             self.roilist.append([roi_left, roi_top, roi_right, roi_bottom])
    #             self.backlist.append([back_left, back_top, back_right, back_bottom])
    #     split_num = len(self.splitlist)
    #     self.batch_num = int(math.ceil(split_num / batch_size)) if batch_size > 0 else 1
    #     self.batch_index = 0
    #     self.preprocess = preprocess if preprocess is not None else lambda x: x
    #     self.lightness_ignore=lightness_ignore

    #     if split_num == 0 : 
    #         raise 

    # def __len__(self):
    #     return self.batch_num

    # def __iter__(self):
    #     return self

    # def __next__(self):
    #     if self.batch_index < self.batch_num:
    #         batch = list()
    #         roilist = list()
    #         backlist = list()
    #         start_idx = self.batch_index * self.batch_size
    #         end_idx = (self.batch_index + 1) * self.batch_size if self.batch_size > 0 else len(self.splitlist)
    #         splits_x_y = self.splitlist[start_idx: end_idx]
    #         roi = self.roilist[start_idx:end_idx]
    #         back = self.backlist[start_idx:end_idx]
    #         for (x, y), (roi_l, roi_t, roi_r, roi_b), (back_l, back_t, back_r, back_b) in zip(splits_x_y, roi, back):
    #             crop = self.img[y:self.crop_h + y, x:self.crop_w + x]
    #             # crop_name = f'C:\\Users\\fxm\\Downloads\\image\\img_dest\\crop\\crop_{x}_{y}.jpg'
    #             # cv2.imwrite(crop_name, crop.copy())
    #             roilist.append([roi_l, roi_t, roi_r, roi_b])
    #             backlist.append([back_l, back_t, back_r, back_b])
    #             batch.append(self.preprocess(crop))
    #         self.batch_index += 1
    #         return batch, roilist, backlist
    #     else:
    #         self.batch_index = 0
    #         raise StopIteration

    def model_inference_get_mask_small(self, inference_func):
        batch, draw_list = self.next()
        result_batch = inference_func(batch)
        return result_batch, draw_list

    def model_inference_get_mask(self, inference_func):
        # with torch.no_grad():
        result_mask = np.zeros((self.h, self.w), dtype=np.float32)
        for batch, roilist, backlist in self:
            result_batch = inference_func(batch)
            for result, (roi_l, roi_t, roi_r, roi_b), (back_l, back_t, back_r, back_b) in zip(result_batch, roilist,
                                                                                                backlist):
                if self.lightness_ignore is not None:
                    if get_lightness(self.img[back_t:back_b, back_l:back_r]) < self.lightness_ignore:
                        continue
                result_mask[back_t:back_b, back_l:back_r] = result[roi_t:roi_b, roi_l:roi_r]
        return result_mask
    
    def get_process_imgs(self):
        col, row = max(self.testlist)
        col, row = col + self.crop_w, row + self.crop_h
        img_tile = np.zeros((row, col, 3))
        img_light = np.zeros((row, col, 3))
        img_diff = np.zeros((row, col, 3))
        for  (crop_x, crop_y), (test_x, test_y) in zip(self.splitlist, self.testlist):
            crop_img = self.img[crop_y:crop_y+self.crop_h, crop_x:crop_x+self.crop_w] 
            crop_img = crop_img.copy()
            img = self.preprocess(crop_img)
            img = img.copy()
            diff_small = img - crop_img
            light_value = get_lightness(crop_img)
            cv2.putText(crop_img, str(int(light_value)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            light_value = get_lightness(img)
            cv2.putText(img, str(int(light_value)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            img_tile[test_y:test_y+self.crop_h, test_x:test_x+self.crop_w] = crop_img
            img_light[test_y:test_y+self.crop_h, test_x:test_x+self.crop_w] = img
            img_diff[test_y:test_y+self.crop_h, test_x:test_x+self.crop_w] = diff_small
        # cv2.imwrite("./tile_img.jpg", img_tile)
        # cv2.imwrite("./add_light_img.jpg", img_light)
        # cv2.imwrite("./diff_img.jpg", img_diff)
        return {"original_img": self.img, "tile_img": img_tile, "add_light_img": img_light, "diff_img": img_diff}


