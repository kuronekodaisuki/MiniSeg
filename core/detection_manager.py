# -*- coding: shift_jis -*-
import os
from os import path
import sys
import copy
import time

import cv2
from queue import Queue
import logging
from logging.handlers import TimedRotatingFileHandler
import tkinter as tk
from tkinter import messagebox

import numpy as np

from core.car_panel_detection import CarPanelDetection
from data.config_data import ConfigData
from data.block_info import BlockInfo
from data.defect_info import DefectInfo
from data.line_process_result import LineProcessResult
from data.common_define import ProcessStatus, ErrorCode, DetectionImageType, ProgressStatus
from data.resource_manager import ResourceManager
from util.directory_util import DirectoryUtil
from util.csv_util import CsvUtil
from util.validators import Validators
# import torch

class DetectionManager:
    def __init__(self):
        self.config_data = ConfigData()
        self.detector = CarPanelDetection()
        self.directory = DirectoryUtil()
        self.csv_file = CsvUtil()
        self.validator = Validators()
        self.resource_man = ResourceManager()

        self.list_img_ext1 = ['.bmp', '.jpg', '.jpeg']
        self.list_img_ext2 = ['.tif', '.tiff', '.jpg', '.jpeg']
        self.logger = logging.getLogger(__name__)
        self.is_stop_process = False
        self.blocks_info_queue = Queue(maxsize=0)
        self.lines_process_status = {}
        self.current_body_number = 0
        self.progress_status = ProgressStatus.INITIALIZE
        self.loading = False
        self.is_first_time = True
        self.init_log_fail = False
        self.WriteLogFail_info = ""
        self.timeout = False
        self.input_imgs = list()

    def initialize(self, config_file_path=None):
        """Args:
            config_file_path (str): The config of detection, directory, etc.
        """
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        err = self.config_data.load_config(config_file_path)
        if err != ErrorCode.NONE:
            handler = TimedRotatingFileHandler("log_file.log", when='midnight')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            reason = self.resource_man.get_string(err)
            self.quit_process(err, reason)

        try:
            handler = TimedRotatingFileHandler(self.config_data.directory_config.log_dest, when='midnight')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config_data.log_level)
        except Exception as e:
            self.init_log_fail = True
            self.progress_status = ProgressStatus.WriteLogFail
            self.WriteLogFail_info = "[" + str(sys._getframe().f_lineno) + "] " + str(e)
            self.show_progress_status(self.WriteLogFail_info)


        '''For test code - start'''
        self.detector.set_safe_logging(self.logger)
        '''For test code - end'''
        success, err_message = self.detector.initialize(self.config_data.model_file_path,
                                                        self.config_data.detection_config)
        if not success:
            self.quit_process(ErrorCode.LOAD_MODEL_FAIL, err_message)

        self.directory.initialize(self.config_data.directory_config)
        self.progress_status = ProgressStatus.Checkable


    def get_all_imgs_files(self):
        os.chdir(self.config_data.directory_config.img_source)
        in_files = os.listdir()
        for in_file in in_files:
            ext = path.splitext(in_file)[1]
            if "E_" in in_file and (ext in self.list_img_ext2 or ext in self.list_img_ext1):
                img_f = path.join(self.config_data.directory_config.img_source, in_file.replace("E_", ""))
                msk_f = path.join(self.config_data.directory_config.img_source, in_file)
                self.input_imgs.append([img_f, msk_f])
    
    def get_all_small_imgs_files(self):
        os.chdir(self.config_data.directory_config.img_source)
        in_files = os.listdir()
        for in_file in in_files:
            ext = path.splitext(in_file)[1]
            if ext in self.list_img_ext2 or ext in self.list_img_ext1:
                img_f = path.join(self.config_data.directory_config.img_source, in_file)
                self.input_imgs.append(img_f)


    def parse_block_info_task(self):
        """
            Scan the image input folder to parse a detection block info
        """
        time_0 = int(round(time.time() * 1000))
        if self.timeout:
            if self.lines_process_status.get(self.current_body_number) is not None:

                line_process_status = self.lines_process_status.get(self.current_body_number)
                lack_blocks = line_process_status.get_lack_image_blocks()
                
                self.__move_csv(self.current_body_number,line_process_status.get_csv_result_file_path())
                #lack_block_str = ("?ｿｽ{?ｿｽf?ｿｽB?ｿｽﾔ搾ｿｽ?ｿｽF%s, ?ｿｽu?ｿｽ?ｿｽ?ｿｽb?ｿｽN?ｿｽﾔ搾ｿｽ?ｿｽF" % self.current_body_number) + \
                #                    ", ".join(str(n) for n in lack_blocks)
                lack_block_str = ("ボディ番号：%s, ブロック番号：" % self.current_body_number) + \
                                    ", ".join(str(n) for n in lack_blocks)
                reason = self.resource_man.get_string(ErrorCode.DETECTION_IMAGE_NOT_FOUND) % lack_block_str
                self.quit_process(ErrorCode.DETECTION_IMAGE_NOT_FOUND, reason, 0)
                self.progress_status = ProgressStatus.Checkable
                self.show_progress_status()
                self.lines_process_status.clear()
                self.current_body_number = 0
                self.timeout = False
        time_1 = int(round(time.time() * 1000))
        self.__safe_logging("parse_block_info_task----1.  runtime(ms)=" + str(time_1 - time_0), 1)

        if self.is_first_time:
            self.show_progress_status()
            self.is_first_time = False

        if not self.is_stop_process:
            os.chdir(self.config_data.directory_config.img_source)
            file_list = os.listdir()
            if not file_list:
                return

            if self.detector.detection_config.debug == 2:
                print("Debugging test:" + "you have 10s to delete test flie") 
                time.sleep(10)
                
            file_list.sort(key=os.path.getctime)

            time_2 = int(round(time.time() * 1000))
            self.__safe_logging("parse_block_info_task----2.  runtime(ms)=" + str(time_2 - time_1), 1)

            detection_images = list()
            detection_area_images = list()
            for file in file_list:
                if not os.path.exists(file):
                    continue
                _, file_ext = os.path.splitext(file)
                list_accept_ext = self.list_img_ext1 + self.list_img_ext2
                if not file.endswith(r".txt") and \
                        file_ext not in list_accept_ext and \
                        not file.endswith(r".csv"):
                    reason = self.resource_man.get_string(ErrorCode.FILE_EXTENSION_INCORRECT) % file
                    self.quit_process(ErrorCode.FILE_EXTENSION_INCORRECT, reason, 0)
                    continue
                
                if file.endswith(r".txt"):
                    # get info from txt file
                    try:

                        file_name_parts = file.split("_")
                        if (len(file_name_parts) != 2) or \
                                (not file_name_parts[0].isdigit()) or \
                                (file_name_parts[1].casefold() != "info.txt"):
                            reason = self.resource_man.get_string(ErrorCode.LINE_INFO_TEXT_NAME_INCORRECT) % file
                            self.quit_process(ErrorCode.LINE_INFO_TEXT_NAME_INCORRECT, reason, 0)
                            continue

                        body_number = int(file_name_parts[0])
                        if len(self.lines_process_status) > 0:
                            continue
                        
                        self.loading = True
                        self.progress_status = ProgressStatus.Checking
                        self.show_progress_status("",body_number)
                        
                        time.sleep(2)
                        info_file = open(file, encoding="utf-8")#shift_jis
                        str_content = info_file.read()
                        str_number = str_content.split(',')

                        if not str_number[1].isdigit():
                            reason = self.resource_man.get_string(ErrorCode.TOTAL_BLOCK_INCORRECT) % file
                            
                            self.quit_process(ErrorCode.TOTAL_BLOCK_INCORRECT, reason, 0)
                            continue

                        info_file.close()
                        deleted_ok = self.directory.delete_file(file)
                        if not deleted_ok:
                            reason = self.resource_man.get_string(ErrorCode.READ_LINE_INFO_FAIL) % file
                            self.quit_process(ErrorCode.READ_LINE_INFO_FAIL, reason, 0)
                            

                        if self.lines_process_status.get(body_number) is None:
                            self.lines_process_status[body_number] = LineProcessResult(int(str_number[1]))
                            self.current_body_number = body_number
                    except Exception as e:
                        self.__safe_logging("[" + str(sys._getframe().f_lineno) + "] " + str(e), 0)
                        reason = self.resource_man.get_string(ErrorCode.READ_LINE_INFO_FAIL) % file
                        self.quit_process(ErrorCode.READ_LINE_INFO_FAIL, reason, 0)
                        continue
                else:
                    try:
                        _, file_ext = os.path.splitext(file)
                        if file_ext not in list_accept_ext:
                            continue
                        if "_結果" in file:
                            continue

                        if file.startswith(r"E"):
                            # example: E_20200219_18802_16_11_1F7_ブロック1_r1
                            err, body_number = self.validator.validate_image_file_name(file,
                                                                                       DetectionImageType.DetectionAreaImageType)
                            if err != ErrorCode.NONE:
                                reason = self.resource_man.get_string(err) % file
                                self.quit_process(err, reason, 0)
                                continue

                            if body_number == self.current_body_number:
                                detection_area_images.append(file)
                        else:
                            # example: 20200219_18802_16_11_1F7_ブロック1_r1
                            err, body_number = self.validator.validate_image_file_name(file, DetectionImageType.DetectionImageType)
                            if err != ErrorCode.NONE:
                                reason = self.resource_man.get_string(err) % file
                                self.quit_process(err, reason, 0)
                                continue

                            if body_number == self.current_body_number:
                                detection_images.append(file)
                    except Exception as e:
                        self.__safe_logging("[" + str(sys._getframe().f_lineno) + "] " + str(e), 0)
                        self.quit_process(ErrorCode.EXCEPTION, str(e))

            time_3 = int(round(time.time() * 1000))
            self.__safe_logging("parse_block_info_task----3. runtime(ms)=" + str(time_3 - time_2), 1)
            try:
                for detection_image in detection_images:
                    self.loading = True
                    # detection_area_image = 'E_' + detection_image
                    detection_area_image = self.__get_detection_area_image_name(detection_image, list_accept_ext)
                    if detection_area_image in detection_area_images:
                        if (not os.path.exists(detection_image)) or (not os.path.exists(detection_area_image)):
                            continue

                        block_body_number = int(detection_image.split('_')[1])
                        line_process_status = self.lines_process_status.get(block_body_number)
                        if line_process_status is not None:
                            block_number = int(detection_image.split('ブロック')[1].split('_')[0])
                            total_blocks = line_process_status.total_blocks
                            is_lack_image_block = line_process_status.is_lack_image_block(block_number)
                            if (block_number <= total_blocks) and is_lack_image_block:
                                block_info = BlockInfo(block_body_number,
                                                       total_blocks,
                                                       block_number,
                                                       detection_image,
                                                       detection_area_image)
                                line_process_status.update_lack_image_blocks(block_number)
                                self.blocks_info_queue.put(block_info)
            except Exception as e:
                self.__safe_logging("[" + str(sys._getframe().f_lineno) + "] " + str(e), 0)
                self.quit_process(ErrorCode.EXCEPTION, str(e))
            time_4 = int(round(time.time() * 1000))
            self.__safe_logging("parse_block_info_task----4. runtime(ms)=" + str(time_4 - time_3), 1)


    def process_detection_task_for_small_image(self):
        no_masking_result_folder_path = path.join(self.config_data.directory_config.img_dest, "result_no_mask")
        os.makedirs(no_masking_result_folder_path, exist_ok=True)
        self.__process_detection_for_small_image(no_masking_result_folder_path)
        
    def process_detection_task(self):
        """
            Main detection function
        """
        total_time_statics = dict()
        total_block = 0
        try:
            for img_info in self.input_imgs:
                time_statics = dict()
                time_1 = int(round(time.time() * 1000))
                # image with area
                masking_result_folder_path = path.join(self.config_data.directory_config.img_dest, "result_with_mask")
                no_masking_result_folder_path = path.join(self.config_data.directory_config.img_dest, "result_no_mask")
                os.makedirs(masking_result_folder_path, exist_ok=True)
                os.makedirs(no_masking_result_folder_path, exist_ok=True)
                detection_result, _, success= self.__process_detection(img_info, no_masking_result_folder_path, masking_result_folder_path, time_statics)
                if not success:
                    self.__safe_logging(f'{path.basename(img_info[0])}  inference failed', 1)
                else:
                    self.__safe_logging(f'{path.basename(img_info[0])}  inference results: len(boxes)={len(detection_result.defect_infos)}', 1)

                time_2 = int(round(time.time() * 1000))
                # time_statics['process_detection_task---1-postprecess runtime(ms)'] = time_2 - time_1
                self.__safe_logging("process_detection_task----1. -postprecess runtime(ms)=" + str(time_2 - time_1), 1)
                # total_time_statics[path.basename[img_info[0]]] = time_statics
            self.loading = False
        except Exception as e:
            self.__safe_logging("[" + str(sys._getframe().f_lineno) + "] " + str(e), 0)
        return total_time_statics, total_block

    def check_timeout(self):
        """
            Check timeout
        """
        
        if len(self.lines_process_status) > 0:
            body_number = self.current_body_number
            lines_process_status_clone = copy.copy(self.lines_process_status)
            if lines_process_status_clone.get(body_number) is not None:
                line_process_status = lines_process_status_clone.get(body_number)
                timeout_exceed, lack_blocks = line_process_status.is_timeout_exceed(self.config_data.timeout)
                if timeout_exceed:
                    # time out, move csv to result folder
                    self.timeout = True
                else:
                    if self.blocks_info_queue.empty() and len(lack_blocks) > 0 and not self.loading:
                        self.progress_status = ProgressStatus.WaitingInputDate    
                        self.show_progress_status()
                    else:
                        if  self.progress_status == ProgressStatus.WaitingInputDate:
                            self.progress_status = ProgressStatus.Checking  


    def __initialize_process_detection(self, cur_line_process: LineProcessResult, cur_block_info: BlockInfo):
        file_name_parts = cur_block_info.detection_image_file_name.split('_')
        cur_line_process.set_files_name(file_name_parts[0],
                                        file_name_parts[1],
                                        self.config_data.directory_config.img_dest)
        csv_file_path = cur_line_process.get_csv_result_file_path()
        err = self.csv_file.create_new_file(csv_file_path)
        if err != ErrorCode.NONE:
            reason = self.resource_man.get_string(ErrorCode.CREATE_CSV_FAIL) % csv_file_path
            self.quit_process(ErrorCode.CREATE_CSV_FAIL, reason, 0)

        folder_path_list = [cur_line_process.get_no_masking_result_folder_path(),
                            cur_line_process.get_masking_result_folder_path()]
        self.__make_result_image_folder(folder_path_list)

    def __make_result_image_folder(self, folder_path_list):
        for folder_path in folder_path_list:
            make_ok = self.directory.make_img_result_folder(folder_path)
            if not make_ok:
                reason = self.resource_man.get_string(ErrorCode.CREATE_OUTPUT_IMAGE_FOLDER_FAIL) % folder_path
                self.quit_process(ErrorCode.CREATE_OUTPUT_IMAGE_FOLDER_FAIL, reason, 0)

    def __process_detection_for_small_image(self, no_mask_result_file_path=""):
        time_0 = int(round(time.time() * 1000))
        self.__safe_logging("推論開始", 1)
        no_mask_result_file_path = no_mask_result_file_path.replace("/", "\\")
        self.detector.detect_small(self.input_imgs, no_mask_result_file_path)

        self.__safe_logging("推論終了", 1)
        time_1 = int(round(time.time() * 1000))
        self.__safe_logging("__process_detection----0. -total runtime(ms)=" + str(time_1 - time_0), 1)



    def __process_detection(self, img_info, no_mask_result_file_path="", mask_result_file_path="", time_statics=None):
        no_mask_result_file_path = no_mask_result_file_path.replace("/", "\\")
        mask_result_file_path = mask_result_file_path.replace('/', '\\')
        success = True
        detection_image_file_name, detection_area_image_file_name = img_info[:]
        file_name_parts = path.basename(detection_image_file_name).split(".")
        no_masking_image_file_name = file_name_parts[0] + "_結果" + "." + file_name_parts[1]
        no_masking_image_result = os.path.join(no_mask_result_file_path, no_masking_image_file_name)

        file_name_parts = path.basename(detection_area_image_file_name).split(".")
        masking_img_file_name = file_name_parts[0] + "_結果" + "." + file_name_parts[1]
        masking_img_result = os.path.join(mask_result_file_path, masking_img_file_name)
        self.__safe_logging("[%s] 画像読み込み" % (img_info[0]), 1)

        try:
            time_0 = int(round(time.time() * 1000))
            detection_image_array = cv2.imdecode(np.fromfile(detection_image_file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            time_1 = int(round(time.time() * 1000))
            time_statics['__process_detection----1-imdecode runtime(ms)']=time_1 - time_0
            self.__safe_logging("__process_detection----1. -imdecode runtime(ms)=" + str(time_1 - time_0), 1)
            if detection_image_array is None:
                reason = self.resource_man.get_string(ErrorCode.CANNOT_READ_IMAGE) % detection_image_file_name
                self.quit_process(ErrorCode.CANNOT_READ_IMAGE, reason, 0)
                success = False
                return DefectInfo(), masking_img_result, success
            
            ''' Do not read the same image repeatedly '''
            output_detection_image_array = None
            if os.path.exists(detection_image_file_name):
                output_detection_image_array = detection_image_array.copy()
            # output_detection_image_array = cv2.imread(detection_image_file_name)
                # output_detection_image_array = cv2.imdecode(np.fromfile(detection_image_file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            time_2 = int(round(time.time() * 1000))
            time_statics['__process_detection----2-imdecode runtime(ms)']=time_2 - time_1
            self.__safe_logging("__process_detection----2. -imdecode runtime(ms)=" + str(time_2 - time_1), 1)

            if output_detection_image_array is None:
                reason = self.resource_man.get_string(ErrorCode.CANNOT_READ_IMAGE) % detection_image_file_name
                self.quit_process(ErrorCode.CANNOT_READ_IMAGE, reason, 0)
                success = False
                return DefectInfo(), masking_img_result, success

            output_mask_image_array = None    
            if os.path.exists(detection_image_file_name):
                output_mask_image_array = detection_image_array.copy()
            # output_mask_image_array is necessary for image to output
                # output_mask_image_array =  cv2.imdecode(np.fromfile(detection_image_file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            time_3 = int(round(time.time() * 1000))
            time_statics['__process_detection----3-imdecode runtime(ms)']=time_3 - time_2
            self.__safe_logging("__process_detection----3. -imdecode runtime(ms)=" + str(time_3 - time_2), 1)
            if output_mask_image_array is None:
                reason = self.resource_man.get_string(ErrorCode.CANNOT_READ_IMAGE) % detection_image_file_name
                self.quit_process(ErrorCode.CANNOT_READ_IMAGE, reason, 0)
                success = False
                return DefectInfo(), masking_img_result, success

            if not os.path.exists(detection_area_image_file_name):
                reason = self.resource_man.get_string(ErrorCode.DETECTION_IMAGE_NOT_FOUND) % detection_area_image_file_name
                self.quit_process(ErrorCode.DETECTION_IMAGE_NOT_FOUND, reason, 0)
                success = False
                return DefectInfo(), masking_img_result, success

            detection_area_image_array = (cv2.imdecode(np.fromfile(detection_area_image_file_name, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) / 255).astype(np.float32)#cv2.imread(detection_area_image_file_name)
            time_4 = int(round(time.time() * 1000))
            time_statics['__process_detection----4-imdecode runtime(ms)']=time_4 - time_3
            self.__safe_logging("__process_detection----4. -imdecode runtime(ms)=" + str(time_4 - time_3), 1)
            if detection_area_image_array is None:
                reason = self.resource_man.get_string(ErrorCode.CANNOT_READ_IMAGE) % detection_area_image_file_name
                self.quit_process(ErrorCode.CANNOT_READ_IMAGE, reason, 0)
                success = False
                return DefectInfo(), masking_img_result, success
        except Exception as e:
            print(e)
            reason = self.resource_man.get_string(ErrorCode.CANNOT_READ_IMAGE) % detection_area_image_file_name
            self.quit_process(ErrorCode.CANNOT_READ_IMAGE, reason, 0)
            success = False
            return DefectInfo(), masking_img_result, success

        # print('__process_masking_detection before convert: ', np.count_nonzero(np.logical_and(detection_area_image_array > 0, detection_area_image_array < 255)))
        # detection_area_image_array = np.where(detection_area_image_array < 255, 0, detection_area_image_array)
        # print('__process_masking_detection after convert: ', np.count_nonzero(np.logical_and(detection_area_image_array > 0, detection_area_image_array < 255)))
        

        self.__safe_logging("[%s] 推論開始" % (img_info[0]), 1)
        time_5 = int(round(time.time() * 1000))
        detection_result, err, err_message, result_image = self.detector.detect(output_detection_image_array, output_mask_image_array, detection_image_array,
                                                                  detection_area_image_array, no_masking_image_result,
                                                                  masking_img_result,time_statics)
        time_6 = int(round(time.time() * 1000))
        time_statics['__process_detection----5--detect runtime(ms)']=time_6 - time_5
        self.__safe_logging("__process_detection----5. -detect runtime(ms)=" + str(time_6 - time_5), 1)
        if (err == ErrorCode.SPLIT_IMAGE_FAIL) or \
                (err == ErrorCode.DRAW_BOUND_FAIL) or \
                (err == ErrorCode.WRITE_RESULT_IMAGE_FAIL):
            if err == ErrorCode.WRITE_RESULT_IMAGE_FAIL:
                reason = self.resource_man.get_string(err) % result_image
            else:
                reason = self.resource_man.get_string(err) % detection_image_file_name
            self.quit_process(err, reason, 0)
            return detection_result, masking_img_result, False
        elif err == ErrorCode.EXCEPTION:
            reason = self.resource_man.get_string(ErrorCode.DETECTION_NO_MASKING_FAIL) % detection_image_file_name
            reason = reason + ("\n(詳細エラー：%s)" % err_message)
            self.quit_process(ErrorCode.DETECTION_NO_MASKING_FAIL, reason, 0)
            return detection_result, masking_img_result, False

        self.__safe_logging("[%s] 推論終了" % (img_info[0]), 1)
        return detection_result, masking_img_result, success


    def __move_csv(self, body_number, csv_file_path):
        self.__safe_logging("[%s][マスク有り] CSV出力" % body_number, 1)
        moved_ok = self.directory.move_file(csv_file_path, self.config_data.directory_config.csv_dest)
        if not moved_ok:
            reason = self.resource_man.get_string(ErrorCode.MOVE_CSV_FAIL) % csv_file_path
            self.quit_process(ErrorCode.MOVE_CSV_FAIL, reason, 0)
        if self.lines_process_status.get(body_number) is not None:
            self.lines_process_status.pop(body_number)

    def __delete_images(self, delete_list):
        for image_file in delete_list:
            deleted_ok = self.directory.delete_file(image_file)
            if not deleted_ok:
                reason = self.resource_man.get_string(ErrorCode.DELETE_IMAGE_FAIL) % image_file
                self.quit_process(ErrorCode.DELETE_IMAGE_FAIL, reason, 0)

    def quit_process(self, error_code, reason="", end_flg=1):
        if end_flg:
            self.is_stop_process = True
        message = reason + "\n(エラーコード： %s)" % error_code.name
        try:
            # write log
            self.__safe_logging(reason, 2)
        except Exception as e:
            message += "\n" + self.resource_man.get_string(
                ErrorCode.WRITE_LOG_FAIL) % self.config_data.directory_config.log_dest

        if end_flg:
            root = tk.Tk()
            root.withdraw()  # 小さなウィンドウを表示させない
            # show popup error
            messagebox.showerror("エラー", message)
            sys.exit(error_code.value)
        pass

    def __check_img_spec(self, block_info, file):
        # Check file extension jpeg or jpg
        # cur_block_info.body_number
        # cur_block_info.block_number
        try:
            _, file_ext = os.path.splitext(file)
            if file_ext not in self.list_img_ext2:
                reason = self.resource_man.get_string(ErrorCode.FILE_EXTENSION_INCORRECT_NOT_TIF)
                reason = self.resource_man.get_string(ErrorCode.FILE_EXTENSION_INCORRECT_NOT_TIF) % (block_info.body_number, block_info.block_number)
                self.quit_process(ErrorCode.FILE_EXTENSION_INCORRECT, reason, 0)
                return False
            
            # Check file size is greater than 150Mb
            size = os.path.getsize(file)
            if size > 157286400:
                reason = self.resource_man.get_string(ErrorCode.FILE_SIZE_ERROR) % (block_info.body_number, block_info.block_number)
                self.quit_process(ErrorCode.FILE_SIZE_ERROR, reason, 0)
                return False
            
            # Check file height and width is greater than 27000px or 2048px
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
            #img = cv2.imread(file)
            height, width, _ = img.shape
            if height > 27000 or width > 2048:
                reason = self.resource_man.get_string(ErrorCode.FILE_SIZE_ERROR) % (block_info.body_number, block_info.block_number)
                self.quit_process(ErrorCode.FILE_SIZE_ERROR, reason, 0)
                return False
            
            return True
        except Exception as e:
            reason = self.resource_man.get_string(ErrorCode.CANNOT_READ_IMAGE) % file
            self.quit_process(ErrorCode.CANNOT_READ_IMAGE, reason, 0)
            return False


    def __get_detection_area_image_name(self, detection_image, accept_ext):
        detection_area_image_name, _ = os.path.splitext(detection_image)
        detection_area_image = 'E_' + detection_image
        for ext in accept_ext:
            detection_area_image = 'E_' + detection_area_image_name + ext
            if (os.path.exists(detection_area_image)):
                return detection_area_image
        return detection_area_image
    
    def __printinfo(self, info):
        print(info)
        
        
    def show_progress_status(self, printinfo = "", body_number = 0):
        if body_number == 0:
            body_number = self.current_body_number
        info = ''
        if self.progress_status == ProgressStatus.Checkable:
            info = '検査可能\n' 
        elif self.progress_status == ProgressStatus.Checking:
            info = '検査中\n'  + '検査車両番号: ' + str(body_number)
        elif self.progress_status == ProgressStatus.WaitingInputDate:
            info = 'データインプット待ち\n' + '検査車両番号: ' + str(body_number)
        elif self.progress_status == ProgressStatus.WriteLogFail:
            info = 'logファイルの書き込みに失敗しました\n' + 'error: ' + printinfo
        
        self.__printinfo(info)
        
    def __safe_logging(self, loginfo = "", logtype = 0):
        # logtype: 0(debug),1(info),2(critical)
        try:
            if self.init_log_fail:
                self.progress_status = ProgressStatus.WriteLogFail          
                self.show_progress_status(self.WriteLogFail_info)
            else:
                if logtype == 0:
                    self.logger.debug(loginfo)
                elif logtype == 1:
                    self.logger.info(loginfo)
                elif logtype == 2:
                    self.logger.critical(loginfo)
  
        except Exception as e:
            self.progress_status = ProgressStatus.WriteLogFail            
            self.show_progress_status("[" + str(sys._getframe().f_lineno) + "] " + str(e))

    def safe_logging(self, loginfo = "", logtype = 0):
            self.__safe_logging(loginfo, logtype)

