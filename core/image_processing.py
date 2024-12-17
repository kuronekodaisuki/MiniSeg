# -*- coding: shift_jis -*-
import numpy as np
import cv2


def split_original_image_to_detection_image(
        image_array: np.array,
        detection_area_image_array: np.array,
        resized_height: int,
        resized_width: int,
):
    """Split original image to small detection image.
    Args:
        image_array: Image array, int HWC mode.
        detection_area_image_array: detection area image
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

    top_ys, left_xs = np.meshgrid(hy, wx)

    split_arrays = list()
    retained_top_lefts = list()

    has_mask = detection_area_image_array.size > 0
    for top_left_y, top_left_x in list(zip(top_ys.flatten(), left_xs.flatten())):
        img_arr = image_array[top_left_y: top_left_y + resized_height, top_left_x: top_left_x + resized_width, :]
        if img_arr.shape[0] == resized_height and img_arr.shape[1] == resized_width:
            if has_mask:
                img_area_arr = detection_area_image_array[top_left_y: top_left_y + resized_height,
                               top_left_x: top_left_x + resized_width, :]
                if img_area_arr.shape[0] == resized_height and img_area_arr.shape[1] == resized_width:
                    if img_area_arr.max() != 255:
                        continue

            if img_arr.max() > 1:
                img_arr = img_arr / 255
            img_arr = img_arr.transpose((2, 0, 1))
            retained_top_lefts.append((top_left_y, top_left_x))
            split_arrays.append(img_arr)

    return retained_top_lefts, np.array(split_arrays)


def post_process_for_original_image(
        mask_area: np.array,
        outputs: np.array,
        top_lefts: list,
        has_mask: False,
        confidence_threshold=0.6,
        detection_size=0):
    """Post process for original image.
    Args:
        output: Model forward inference result.
        top_lefts: The position of the split small image on the original image.
        confidence_threshold：Threshold to control the number of candidate frames.
                      The larger the threshold, the smaller the number of boxes.
        detection_size: Check size of error area.
    returns:
        predict_boxes: The detect result above the original image.
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
                # if has_mask:
                #     x1 = x + top_lefts[idx][1] + round(w / 2)
                #     y1 = y + top_lefts[idx][0] + round(h / 2)
                #     if not (mask_area[y1][x1][0] == 255 and mask_area[y1][x1][1] == 255 and mask_area[y1][x1][2] == 255):
                #         continue

                det_box_xmin = x - 3 + top_lefts[idx][1]
                det_box_ymin = y - 3 + top_lefts[idx][0]
                det_box_xmax = x + w + 3 + top_lefts[idx][1]
                det_box_ymax = y + h + 3 + top_lefts[idx][0]

                if has_mask:
                    x1 = round((det_box_xmin + det_box_xmax) / 2)
                    y1 = round((det_box_ymin + det_box_ymax) / 2)
                    if not (mask_area[y1][x1][0] == 255 and mask_area[y1][x1][1] == 255 and mask_area[y1][x1][2] == 255):
                        continue
                
                # 長辺チェック
                if h > w:
                    w = h
                if w >= detection_size:
                    predict_boxes.append((det_box_xmin, det_box_ymin, det_box_xmax, det_box_ymax))
    return predict_boxes
