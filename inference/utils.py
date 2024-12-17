# -*- coding: shift_jis -*-
# @Time : 2023/2/1 15:37
# @Author : Linyi Zuo
# @Project : video-deep-fusion
# @File : spliter.py
# @Software: PyCharm
# import torch
import numpy as np
import random
import base64

image_ext = ['.png', '.bmp', '.jpg']


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


'''
Description: 
Author: Yongliang Lan
Date: 2022-12-14 21:24:23
LastEditTime: 2022-12-14 22:34:16
email: yongliang.lan@thundersoft.com
'''
import json
import os.path

import cv2

# import torch
import numpy as np
# import onnxruntime
# from torch.nn import DataParallel

# CUDA_AVAILABLE = torch.cuda.is_available()


def get_lightness(image_bgr):
    return np.mean(np.max(image_bgr, axis=2))


def get_std(image_bgr, round_num=2):
    std = round(float(np.mean(cv2.meanStdDev(image_bgr)[1])), round_num)
    return std


def split_original_image_to_detection_image(
        image_array: np.array,
        resized_height: int,
        resized_width: int
):
    """Split original image to small detection image.
    Args:
        image_array: Image array, int HWC mode.
        resized_height: Split height size for original image.
        resized_width: Split width size for original image.
    returns:
        top_lefts: The position of the split small image on the original image.
        split_arrays: Split small images array with preprocessed.
    """
    origin_height, origin_width, _ = image_array.shape
    assert origin_height > resized_height and origin_width > resized_width, 'The input image is too small.'
    hy = np.arange(0, origin_height, resized_height)
    wx = np.arange(0, origin_width, resized_width)

    # print(origin_height, origin_width)
    # print(hy)
    # print(wx)

    top_ys, left_xs = np.meshgrid(hy, wx)

    split_arrays = list()
    retained_top_lefts = list()
    for top_left_y, top_left_x in list(zip(top_ys.flatten(), left_xs.flatten())):
        img_arr = image_array[top_left_y: top_left_y + resized_height,
                  top_left_x: top_left_x + resized_width, :]
        # if not img_arr.shape == (96,96,3):
        # print(img_arr.shape)
        if img_arr.shape[0] == resized_height and img_arr.shape[1] == resized_width:
            if img_arr.max() > 1:
                img_arr = img_arr / 255
            img_arr = img_arr.transpose((2, 0, 1))
            retained_top_lefts.append((top_left_y, top_left_x))
            split_arrays.append(img_arr)

    return retained_top_lefts, np.array(split_arrays)


def post_process_for_original_image(
        outputs: np.array,
        top_lefts: list,
        confidence_threshold=0.6,
        detection_size=0
):
    """Post processes for original image.
    Args:
        output: Model forward inference result.
        top_lefts: The position of the split small image on the original image.
        confidence_threshold：Threshold to control the number of candidate frames.
                      The larger the threshold, the smaller the number of boxes.
    returns:
        predict_boxes: Detect result above the original image.
    """
    predict_boxes = list()
    for idx in range(outputs.shape[0]):
        full_mask = outputs[idx, 0, :, :]
        mask = full_mask > confidence_threshold
        reconstruction_image = (mask * 255).astype(np.uint8)

        if np.count_nonzero(reconstruction_image) < 30:
            continue

        contours, _ = cv2.findContours(reconstruction_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                det_box_xmin = x - 3 + top_lefts[idx][1]
                det_box_ymin = y - 3 + top_lefts[idx][0]
                det_box_xmax = x + w + 3 + top_lefts[idx][1]
                det_box_ymax = y + w + 3 + top_lefts[idx][0]
                if abs(det_box_ymax - det_box_ymin) >= detection_size:
                    predict_boxes.append((det_box_xmin, det_box_ymin, det_box_xmax, det_box_ymax))
                # x, y, w, h = cv2.boundingRect(cnt)
                # det_box_xmin = x - 3 + top_lefts[idx][1]
                # det_box_ymin = y - 3 + top_lefts[idx][0]
                # det_box_xmax = x + w + 3 + top_lefts[idx][1]
                # det_box_ymax = y + w + 3 + top_lefts[idx][0]
                # predict_boxes.append((det_box_xmin, det_box_ymin, det_box_xmax, det_box_ymax))
    return predict_boxes


def draw_boxes_on_image(image, boxes, is_gt_box=True, color_type=None):
    if not color_type:
        if is_gt_box:
            color_type = (255, 0, 0)
        else:
            color_type = (0, 0, 255)

    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color_type, 1)
    return image


# def load_model(args, checkpoint):
#     if checkpoint.endswith(".onnx"):
#         raise NotImplementedError
#         # model = onnxruntime.InferenceSession(
#         #     checkpoint,
#         #     providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
#         # )
#         # if CUDA_AVAILABLE:
#         #     model.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
#     elif checkpoint.endswith(".pth"):
#         dev = torch.device("cuda") if CUDA_AVAILABLE else torch.device("cpu")
#         if args.net == "miniseg":
#             from model.miniseg.miniseg import MiniSeg
#             model = MiniSeg(is_training=False)
#         # model = torch.nn.DataParallel(MiniSeg(is_training=True))
#         ckpt = torch.load(checkpoint, map_location=dev)
#         # if 'module.' in ckpt.keys().__next__():
#         # Remove "module." e.g. module.predict3.1.weight -> predict3.1.weight
#         sd = {k[7:]: v for k, v in ckpt.items()}
#         # elif isinstance(ckpt, dict):
#         #     sd = ckpt
#         # else:
#         #     raise Exception("Unsupported Checkpoint.")
#         model.load_state_dict(sd)
#         model.to(dev)
#     else:
#         raise Exception("Invalid checkpoint. Only .onnx and .pth are supported.")

#     return model


def model_eval(model, small_images):
    """

    :param model: InferenceSession or torch.nn.Module.
    :param small_images: array, shape=(n_image, C, H, W)

    :return array(shape=(n_image, 1, H, W)), probs of each pixel.
    """
    # if isinstance(model, onnxruntime.InferenceSession):
    #     batch = {model.get_inputs()[0].name: small_images.detach().cpu().numpy().astype(np.float32)}
    #     # logits: list(len=4 ?)[array(shape=(n_crop, 1, H, W))]
    #     logits = model.run(None, batch)
    #     # outputs: array(shape=(n_crop, 1, H, W))
    #     outputs = 1 / (1 + np.exp(-logits[0]))
    if isinstance(model, torch.nn.Module):
        model.eval()
        dev = next(model.parameters()).device
        batch = torch.asarray(small_images)
        batch = batch.to(device=dev, dtype=torch.float32)

        with torch.no_grad():
            # logits: list(len=4)[array(shape=(n_crop, 1, H, W))] ?
            logits = model(batch)[0]
            # print(f"---- logits: {len(logits)}, {logits[0].shape}")
            # outputs: array(shape=(n_crop, 1, H, W))
            outputs = torch.sigmoid(logits).cpu().numpy()
    else:
        raise Exception('UNSUPPORTED model format!!')

    return outputs


def calcuate_iou(box1, box2, epsilon=1e-5):
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


def calculate_fp(gt_bboxes, pred_bboxes, iou_thr=0.5):
    if len(pred_bboxes) == 0:
        fp_bboxes = []
        tp, fp, fn = 0, 0, len(gt_bboxes)
        return fp, fp_bboxes

    if len(gt_bboxes) == 0:
        fp = len(pred_bboxes)
        fp_bboxes = pred_bboxes
        return fp, fp_bboxes

    tp, fp, fn = 0, 0, 0

    gt_hit_flag = [0 for _ in range(len(gt_bboxes))]
    fp_bboxes_list = []
    for bbox_pre in pred_bboxes:
        x1, y1, x2, y2 = bbox_pre
        if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
            continue
        hit_gt = False

        for index, bbox_gt in enumerate(gt_bboxes):
            iou = calcuate_iou(bbox_gt, bbox_pre)
            if iou > iou_thr:
                gt_hit_flag[index] = 1
                hit_gt = True
        if hit_gt:
            tp += 1
        if not hit_gt:
            fp += 1
            fp_bboxes_list.append(bbox_pre)
    return fp, fp_bboxes_list


def calculate_tp_fn(gt_bboxes, pred_bboxes, iou_thr=0.5):
    if len(pred_bboxes) == 0:
        tp = 0
        tp_bboxes = []
        fn = len(gt_bboxes)
        fn_bboxes = gt_bboxes
        return tp, fn, tp_bboxes, fn_bboxes

    if len(gt_bboxes) == 0:
        tp = 0
        tp_bboxes = []
        fn = 0
        fn_bboxes = []
        return tp, fn, tp_bboxes, fn_bboxes

    tp, fp, fn = 0, 0, 0
    my_fn = 0
    tp_bboxes = []
    fn_bboxes = []
    gt_hit_flag = [0 for _ in range(len(gt_bboxes))]
    for index, bbox_gt in enumerate(gt_bboxes):
        hit_gt = False
        area_gt = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1])

        for bbox_pre in pred_bboxes:
            x1, y1, x2, y2 = bbox_pre
            if abs(x2 - x1) < 1 or abs(y2 - y1) < 1:
                continue

            iou = calcuate_iou(bbox_pre, bbox_gt)
            if iou > iou_thr:
                gt_hit_flag[index] = 1
                hit_gt = True
                tp_bboxes.append(bbox_pre)
            if hit_gt:
                break
        if hit_gt:
            tp += 1
        else:
            my_fn += 1
            fn_bboxes.append(bbox_gt)
    fn = gt_hit_flag.count(0)
    assert fn == my_fn

    return tp, fn, tp_bboxes, fn_bboxes


def get_enclosing_rect(mask, is_gt=True):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, 96)
        y2 = min(y + h, 96)
        if not is_gt:
            if (w < 5) and (h < 5):
                continue

        bboxes.append([x1, y1, x2, y2])
    return bboxes


def calculate_pixel_iou(mask, pred, epsilon=1e-5):
    intersection = torch.sum(torch.logical_and(mask, pred))

    union = torch.sum((torch.logical_or(pred, mask)))
    # Intersection over Union
    iou_score = intersection / (union + epsilon)
    # print("intersection:{} union:{}".format(intersection, union))

    return iou_score


def image_path_to_json(path):
    return os.path.splitext(path)[0] + '.json'

def imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img
class LabelmeJson:

    def __init__(self, version='5.0.5'):
        self.shapes = list()

    def add_polygon(self, labelname: str, points):
        self.shapes.append(
            {
                'label': labelname,
                'points': points.tolist(),
                'group_id': None,
                'shape_type': 'polygon',
                'flags': {}
            }
        )

    def save(self, json_path, image_path):
        img = imread(image_path)
        h, w = img.shape[:2]
        with open(image_path, 'rb') as f:
            data = f.read()
            b64 = base64.b64encode(data)
        save_dict = {
            "version": "5.0.5",
            "flags": {},
            "shapes": self.shapes,
            'imagePath': os.path.basename(image_path),
            'imageData': str(b64)[2:-1],
            "imageHeight": h,
            "imageWidth": w
        }
        with open(json_path, 'w', encoding='shift_jis') as f:
            json.dump(save_dict, f)

def addto180(image_bgr: np.ndarray):
    # print('mean________')
    l = get_lightness(image_bgr)
    up = 180.1 - l
    if up < 0:
        return image_bgr
    result = image_bgr.copy().astype(np.float32)
    result += up
    result[result > 255] = 255
    result[result < 0] = 0
    result = result.astype(np.uint8)
    return result

class JsonLabel:
    def __init__(self, json_path):
        if os.path.splitext(json_path)[1] != '.json':
            json_path = image_path_to_json(json_path)
            assert os.path.exists(json_path)
        with open(json_path, 'r', encoding='shift_jis') as f:
            self.json_data = json.load(f)
        pts = list()
        boxes = list()
        for shapeinfo in self.json_data['shapes']:
            assert shapeinfo['shape_type'] == 'polygon'
            points = np.round(np.array(shapeinfo['points'])).astype(int)
            x, y, w, h = cv2.boundingRect(points)
            boxes.append([x, y, x + w, y + h])
            pts.append(points)
        self.polygon_points: list = pts
        self.boxes: list = boxes
def list_image_folder(data_root):
    label_dir_name = '欠陥画像'
    image_dir_name = '全体画像'
    path_stack = [data_root]
    image_folder_list = list()

    def search():
        path = os.path.join(*path_stack)
        if os.path.isdir(path):
            for folder in os.listdir(path):
                if folder == image_dir_name:
                    image_folder = os.path.join(path, image_dir_name)
                    label_folder = os.path.join(path, label_dir_name)
                    if not os.path.exists(label_folder):
                        label_folder = None
                    image_folder_list.append([image_folder, label_folder])
                elif folder != label_dir_name:
                    path_stack.append(folder)
                    search()
        path_stack.pop()

    search()
    return image_folder_list

def list_image_any(data_root):
    path_stack = [data_root]
    image_list = list()
    def search():
        path = os.path.join(*path_stack)
        if os.path.isdir(path):
            for folder in os.listdir(path):
                path_stack.append(folder)
                search()
        else:
            name, ext = os.path.splitext(path)
            if ext in image_ext:
                image_list.append(path)
        path_stack.pop()
    search()
    return image_list

def list_image(data_root):
    ret = list()
    for imgfolder,_ in list_image_folder(data_root):
        for file in os.listdir(imgfolder):
            if os.path.splitext(file)[1] in image_ext:
                ret.append(os.path.join(imgfolder, file))
    return ret
