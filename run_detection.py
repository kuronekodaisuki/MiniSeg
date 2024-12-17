# -*- coding: shift_jis -*-
import time
from threading import Timer
from core.detection_manager import DetectionManager
from data.common_define import ErrorCode
import sys
import os
from util.excel_util import generate_excel

if __name__ == '__main__':
    """Step 1: Load configs and initialize"""
    config_file_path = "config/configs.json"
    """Initialize logger"""
    print("Start process.......")
    detect_mng = DetectionManager()
    detect_mng.initialize(config_file_path)
    # detect_mng.get_all_imgs_files()
    # total_time_statics, total_blocks = detect_mng.process_detection_task()
    detect_mng.get_all_small_imgs_files()
    detect_mng.process_detection_task_for_small_image()

    print("All Done")
    
    # pip list --format=freeze >requirement.txt
