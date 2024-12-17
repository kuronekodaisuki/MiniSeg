# -*- coding: shift_jis -*-
import cv2

from data.common_define import ErrorCode

import onnxruntime
import numpy as np
import os

from core.image_processing import split_original_image_to_detection_image, post_process_for_original_image
from data.defect_info import DefectInfo
from data.detection_config import DetectionConfig
from inference.postprocess import PostProcess
from inference.spliter import Spliter
from inference.infer import Inference
from inference.transforms import normalize,AddLightness
import time

class CarPanelDetection:
    """Detection Method"""

    def __init__(self):
        self.detect_model = None
        self.detection_config = DetectionConfig(0.6, 96, 96, False, 0)
        pass

    @property
    def get_algo_info(self):
        """Require algorithm info.
        Returns:
            Algorithm info (version) str.
        """
        return 'version_1.0'

    def initialize(self, model_file_path=None, detection_config=None):
        """Args:
            model_file_path (str): Input model weights file path
            detection_config (DetectionConfig): The config of detection. The config contains confidence
            threshold, some hyperparameters, etc.
        """
        try:
            self.detect_model = onnxruntime.InferenceSession(model_file_path, providers=['CUDAExecutionProvider', "CPUExecutionProvider"],opset_version=11)
            self.detection_config.confidence_threshold = detection_config.confidence_threshold
            self.detection_config.resized_height = detection_config.resized_height
            self.detection_config.resized_width = detection_config.resized_width
            self.detection_config.draw_result = detection_config.draw_result
            self.detection_config.detection_size = detection_config.detection_size
            self.detection_config.debug = detection_config.debug
            return True, ""
        except Exception as e:
            return False, str(e)
    

    def detect_small(self, img_list, out_path):
        time_0 = int(round(time.time() * 1000))
        defect_info = DefectInfo()
        try:
            # new need begin
            post_processor = PostProcess(ignore=3, connect=3, area_ignore=32)
            infer = Inference(self.detect_model, normalize, 0)
            preprocess = AddLightness(86, 180)
            split_processor = Spliter(img_list, preprocess, 10)

            # new need end
        except Exception as e:
            return defect_info, ErrorCode.SPLIT_IMAGE_FAIL, str(e), ""
        
        # try:
        assert(not self.detection_config.debug == 4)
        # new need begin
        result_mask, draw_list = split_processor.model_inference_get_mask_small(infer.list_inference)
        time_1 = int(round(time.time() * 1000))
        self.safe_logging("detect----1. -read images, preprocess(matrix) and inference runtime(ms)=" + str(time_1 - time_0), 1)
        predict_boxes_list = post_processor.get_bbox_for_small(result_mask, self.detection_config.confidence_threshold)

        defect_info.defect_infos = predict_boxes_list
        time_2 = int(round(time.time() * 1000))
        self.safe_logging("detect----2. -post-process filter bounding box runtime(ms)=" + str(time_2 - time_1), 1)

        # except Exception as e:
        #     return defect_info, ErrorCode.EXCEPTION, str(e), ""
        
        if self.detection_config.draw_result:
            time_4_0 = int(round(time.time() * 1000))
            i = 0
            for predict_boxes in predict_boxes_list:
                out_img_f = os.path.join(out_path, os.path.basename(img_list[i]))
                out_img = draw_list[i]
                for predict_box in (predict_boxes):
                    cv2.rectangle(out_img, (int(predict_box[0]) - 30, int(predict_box[1]) - 30),
                                (int(predict_box[2]) + 30, int(predict_box[3]) + 30), (0, 255, 0), 3)
                result, encoded_img = cv2.imencode('.jpg', out_img)
                encoded_img.tofile(out_img_f)
                if not result:
                    self.safe_logging(f"detect----WRITE_RESULT_IMAGE_FAIL: {out_img_f}", 1)
                i += 1

            time_4 = int(round(time.time() * 1000))
            self.safe_logging("detect----3. -WRITE_RESULT_IMAGE runtime(ms)=" + str(time_4 - time_4_0), 1)


    def detect(self, output_img_array, output_mask_img_array, detection_img_array, detection_area_img_array, path_img, path_mask_img, time_statics):
        """Car panel defect detect for single image.
        Args:
            output_img_array (numpy.array): [H,W,3] BGR array.
            output_mask_img_array (numpy.array): [H,W,3] BGR array.
            detection_img_array (numpy.array): [H,W,3] BGR array
            detection_area_img_array (numpy.array): [H,W,3] BGR array
            path_img: The result image file path
            path_mask_img: The result image file path

        Returns:
            DefectInfo object.
        """
        defect_info = DefectInfo()
        try:
            # new need begin
            time_0 = int(round(time.time() * 1000))
            post_processor = PostProcess(ignore=3, connect=3, area_ignore=32)
            infer = Inference(self.detect_model, normalize, 0)
            preprocess = AddLightness(86, 180)
            split_processor = Spliter(detection_img_array, self.detection_config.resized_height, self.detection_config.resized_width, 8, True, 0, preprocess, 10)
            time_1 = int(round(time.time() * 1000))
            time_statics['detect----0-prepare data list runtime(ms)']=time_1 - time_0
            self.safe_logging("detect----0. -prepare data list runtime(ms)=" + str(time_1 - time_0), 1)
            # new need end
        except Exception as e:
            return defect_info, ErrorCode.SPLIT_IMAGE_FAIL, str(e), ""

        try:

            assert(not self.detection_config.debug == 4)
            # new need begin
            result_mask = split_processor.model_inference_get_mask(infer.list_inference)
            time_1_1 = int(round(time.time() * 1000))
            time_statics['detect----1-preprocess(crop and matrix) and inference runtime(ms)']=time_1_1 - time_1
            self.safe_logging("detect----1. -preprocess(crop and matrix) and inference runtime(ms)=" + str(time_1_1 - time_1), 1)
            result_mask_ = result_mask
            result_mask_ = (result_mask_ > self.detection_config.confidence_threshold).astype(np.uint8)
            predict_boxes ,_ = post_processor.get_bbox(result_mask_)

            filter_mask = detection_area_img_array #(cv2.imdecode(np.fromfile(detection_area_img_array, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) / 255).astype(np.float32)
            fh = filter_mask.shape[0]
            rh = result_mask.shape[0]

            if fh > rh:
                result_mask = filter_mask[:rh] * result_mask
            else:
                result_mask[:fh] = filter_mask * result_mask[:fh]

            result_mask = (result_mask > self.detection_config.confidence_threshold).astype(np.uint8)
            mask_predict_boxes, _ = post_processor.get_bbox(result_mask)

            defect_info.defect_infos = mask_predict_boxes
            time_2 = int(round(time.time() * 1000))
            time_statics['detect----2-post-process filter bounding box runtime(ms)']=time_2 - time_1_1
            self.safe_logging("detect----2. -post-process filter bounding box runtime(ms)=" + str(time_2 - time_1_1), 1)

        except Exception as e:
            return defect_info, ErrorCode.EXCEPTION, str(e), ""

        if self.detection_config.draw_result:

            if self.detection_config.debug == 7:
                name = os.path.basename(path_img)
                path = os.path.dirname(path_img)
                print(name,path)
                name = name.split('.')[0] 
                list = split_processor.get_process_imgs()
                for key,value in list.items():
                    print(path + "/" + name + "_" + key + ".png")
                    result = cv2.imencode('.png', value)[1].tofile(path + "/" + name + "_" + key + ".png")
                    #result = cv2.imwrite(path + "/" + name + "_" + key + ".png", value)
                    continue
                time_3 = int(round(time.time() * 1000))
                time_statics['detect----3-debug==7 and write png runtime(ms)']=time_3 - time_2
                self.safe_logging("detect----3. -debug == 7 and write png runtime(ms)=" + str(time_3 - time_2), 1)
                    
            try:
                assert(not self.detection_config.debug == 3)
                for predict_box in (predict_boxes):
                    cv2.rectangle(output_img_array, (int(predict_box[0]) - 30, int(predict_box[1]) - 30),
                                  (int(predict_box[2]) + 30, int(predict_box[3]) + 30), (0, 255, 0), 3)
            except Exception as e:
                return defect_info, ErrorCode.DRAW_BOUND_FAIL, str(e), ""
        
            
            try:
                time_4_0 = int(round(time.time() * 1000))
                result, encoded_img = cv2.imencode('.jpg', output_img_array)
                encoded_img.tofile(path_img)
                time_4 = int(round(time.time() * 1000))
                time_statics['detect----4-WRITE_RESULT_IMAGE_FAIL and write png runtime(ms)']=time_4 - time_4_0
                self.safe_logging("detect----4. -WRITE_RESULT_IMAGE_FAIL and write png runtime(ms)=" + str(time_4 - time_4_0), 1)
                if not result:
                    return defect_info, ErrorCode.WRITE_RESULT_IMAGE_FAIL, str(e), path_img
            except Exception as e:
                return defect_info, ErrorCode.WRITE_RESULT_IMAGE_FAIL, str(e), path_img

            try:
                assert(not self.detection_config.debug == 3)
                for predict_box in (mask_predict_boxes):
                    cv2.rectangle(output_mask_img_array, (predict_box[0] - 30, predict_box[1] - 30),
                                  (predict_box[2] + 30, predict_box[3] + 30), (0, 255, 0), 3)
            except Exception as e:
                return defect_info, ErrorCode.DRAW_BOUND_FAIL, str(e), ""

            try:
                time_5_0 = int(round(time.time() * 1000))
                result, encoded_img = cv2.imencode('.jpg', output_mask_img_array)
                encoded_img.tofile(path_mask_img)
                time_5 = int(round(time.time() * 1000))
                time_statics['detect----5-WRITE_RESULT_IMAGE_FAIL and write JPG runtime(ms)']=time_5 - time_5_0
                self.safe_logging("detect----5. -WRITE_RESULT_IMAGE_FAIL and write JPG runtime(ms)=" + str(time_5 - time_5_0), 1)
                #result = cv2.imwrite(path_mask_img, output_mask_img_array)
                if not result:
                    return defect_info, ErrorCode.WRITE_RESULT_IMAGE_FAIL, str(e), path_mask_img
            except Exception as e:
                return defect_info, ErrorCode.WRITE_RESULT_IMAGE_FAIL, str(e), path_mask_img


        return defect_info, ErrorCode.NONE, "", ""

    def deinitialize(self):
        """Release resource
        Returns:
            None
        """
        self.detect_model = None
    
    def set_safe_logging(self, logger):
        self.logger = logger

    def safe_logging(self, loginfo = "", logtype = 0):
        if logtype == 0:
            self.logger.debug(loginfo)
        elif logtype == 1:
            self.logger.info(loginfo)
        elif logtype == 2:
            self.logger.critical(loginfo)

