# -*- coding: shift_jis -*-
import datetime
from data.block_process_result import BlockProcessResult
from data.common_define import ProcessStatus


class LineProcessResult(object):
    """Structure of the process result for a line with multiblock"""
    def __init__(self, total_blocks):
        self.status = ProcessStatus.INITIALIZE
        self.total_blocks = total_blocks
        self.done_blocks = 0
        self.blocks = [BlockProcessResult(i) for i in range(1, total_blocks+1)]
        self.lack_image_blocks = [i for i in range(1, total_blocks+1)]
        self.csv_result_file_path = ""
        self.csv_lineno = 0
        self.no_masking_result_folder_path = ""
        self.masking_result_folder_path = ""
        self.process_started_at = datetime.datetime.now()

    def __iter__(self):
        return iter(self.blocks)
    
    def get_status(self):
        return self.status

    def get_csv_result_file_path(self):
        return self.csv_result_file_path
    
    def get_no_masking_result_folder_path(self):
        return self.no_masking_result_folder_path
    
    def get_masking_result_folder_path(self):
        return self.masking_result_folder_path
    
    def get_csv_lineno(self):
        return self.csv_lineno

    def get_lack_image_blocks(self):
        return self.lack_image_blocks

    def set_files_name(self, date_str="", body_number="", path_result=""):
        self.csv_result_file_path = body_number + "_" +  date_str + ".csv"
        self.csv_lineno = 0
        result_folder_path = path_result + "/" + date_str[0:4] + "年" + date_str[4:6] + "月" + date_str[6:8] + "日"
        result_folder_path = result_folder_path + "/" + body_number
        self.no_masking_result_folder_path = result_folder_path + "/" + "マスク無し"
        self.masking_result_folder_path = result_folder_path + "/" + "マスク有り"

    def update_line_process_result(self, block_number, no_added_lines):
        self.blocks[block_number-1].is_completed = True
        self.done_blocks = self.done_blocks + 1
        if self.done_blocks < self.total_blocks:
            self.status = ProcessStatus.IN_PROGRESS
        else:
            self.status = ProcessStatus.DONE
        
        self.csv_lineno = self.csv_lineno + no_added_lines

    def update_lack_image_blocks(self, block_number):
        index = 0
        for number in self.lack_image_blocks:
            if number == block_number:
                self.lack_image_blocks.pop(index)
                break
            index += 1

    def is_timeout_exceed(self, timeout):
        if len(self.lack_image_blocks) > 0:
            now = datetime.datetime.now()
            total_process_time = now - self.process_started_at
            if total_process_time.total_seconds() > timeout:
                return True, self.lack_image_blocks
        return False, self.lack_image_blocks

    def is_lack_image_block(self, block_number):
        return block_number in self.lack_image_blocks
