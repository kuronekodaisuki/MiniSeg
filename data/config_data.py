# -*- coding: shift_jis -*-
import json
import os

from data.common_define import ErrorCode, LogLevel
from data.detection_config import DetectionConfig
from data.directory_config import DirectoryConfig
from util.validators import Validators


class ConfigData(object):
    """Structure of config data"""

    def __init__(self):
        self.model_file_path = ""
        self.timeout = 600
        self.detection_config = DetectionConfig(0.6, 96, 96, False, 0)
        self.directory_config = DirectoryConfig()
        self.log_level = 20
        pass

    def load_config(self, config_file_path):
        try:
            with open(config_file_path, 'r', encoding='utf-8') as fp_config:
                configs = json.load(fp_config)
                self.model_file_path = configs['model_file_path']
                if not os.path.exists(self.model_file_path):
                    return ErrorCode.MODEL_NOT_EXIST

                if isinstance(configs['timeout'], int):
                    self.timeout = int(configs['timeout'])

                if not isinstance(configs['detection_config']['confidence_threshold'], float):
                    return ErrorCode.DETECTION_CONFIG_INVALID
                self.detection_config.confidence_threshold = float(configs['detection_config']['confidence_threshold'])

                if not isinstance(configs['detection_config']['resized_height'], int):
                    return ErrorCode.DETECTION_CONFIG_INVALID
                self.detection_config.resized_height = int(configs['detection_config']['resized_height'])

                if not isinstance(configs['detection_config']['resized_width'], int):
                    return ErrorCode.DETECTION_CONFIG_INVALID
                self.detection_config.resized_width = int(configs['detection_config']['resized_width'])

                if not isinstance(configs['detection_config']['draw_result'], int):
                    return ErrorCode.DETECTION_CONFIG_INVALID
                self.detection_config.draw_result = int(configs['detection_config']['draw_result'])

                if not isinstance(configs['detection_config']['detection_size'], int):
                    return ErrorCode.DETECTION_CONFIG_SIZE_ERROR
                detection_size = int(configs['detection_config']['detection_size'])
                if detection_size < 0 or detection_size > 96:
                    return ErrorCode.DETECTION_CONFIG_SIZE_INVALID_VALUE
                self.detection_config.detection_size = detection_size

                self.directory_config.img_source = configs['directory_config']['img_source']
                if not os.path.isdir(self.directory_config.img_source):
                    return ErrorCode.INPUT_IMAGE_FOLDER_PATH_INCORRECT

                self.directory_config.img_dest = configs['directory_config']['img_dest']
                if not os.path.isdir(self.directory_config.img_dest):
                    return ErrorCode.OUTPUT_IMAGE_FOLDER_PATH_INCORRECT

                self.directory_config.csv_dest = configs['directory_config']['csv_dest']
                if not os.path.isdir(self.directory_config.csv_dest):
                    return ErrorCode.CSV_FOLDER_PATH_INCORRECT

                self.directory_config.log_dest = configs['directory_config']['log_dest']
                log_fname = os.path.split(self.directory_config.log_dest)
                if not os.path.isdir(log_fname[0]) or not log_fname[1].endswith('.log'):
                    return ErrorCode.LOG_FOLDER_PATH_INCORRECT

                if isinstance(configs['log_level'], int):
                    log_level = int(configs['log_level'])
                    if log_level in LogLevel._value2member_map_:
                        self.log_level = log_level

                # for debug  test
                try:
                    if not isinstance(configs['detection_config']['debug'], int):
                        return ErrorCode.DETECTION_CONFIG_INVALID
                    self.detection_config.debug = int(configs['detection_config']['debug'])
                except Exception as e:
                    self.detection_config.debug = 0

            return ErrorCode.NONE
        except Exception as e:
            return ErrorCode.LOAD_CONFIG_FAIL
