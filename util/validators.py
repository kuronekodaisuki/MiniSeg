# -*- coding: shift_jis -*-
import datetime
import errno, os
import sys

from data.common_define import DetectionImageType, ErrorCode


class Validators:
    """Validate setting data and input data"""
    def __init__(self):
        self.underscore_count_map = {
            DetectionImageType.DetectionImageType: 7,
            DetectionImageType.DetectionAreaImageType: 8}

        self.datetime_start_index_map = {
            DetectionImageType.DetectionImageType: 0,
            DetectionImageType.DetectionAreaImageType: 1}

        self.body_number_start_index_map = {
            DetectionImageType.DetectionImageType: 1,
            DetectionImageType.DetectionAreaImageType: 2}

        self.block_number_start_index_map = {
            DetectionImageType.DetectionImageType: 5,
            DetectionImageType.DetectionAreaImageType: 6}

        self.image_format_incorrect_error_map = {
            DetectionImageType.DetectionImageType: ErrorCode.DETECTION_IMAGE_FORMAT_INCORRECT,
            DetectionImageType.DetectionAreaImageType: ErrorCode.DETECTION_AREA_IMAGE_FORMAT_INCORRECT}

        self.datetime_error_map = {
            DetectionImageType.DetectionImageType: ErrorCode.DETECTION_IMAGE_DATETIME_INCORRECT,
            DetectionImageType.DetectionAreaImageType: ErrorCode.DETECTION_AREA_IMAGE_DATETIME_INCORRECT}

        self.body_number_error_map = {
            DetectionImageType.DetectionImageType: ErrorCode.DETECTION_IMAGE_BODY_NUMBER_INCORRECT,
            DetectionImageType.DetectionAreaImageType: ErrorCode.DETECTION_AREA_IMAGE_BODY_NUMBER_INCORRECT}
        pass

    @staticmethod
    def is_pathname_valid(pathname: str) -> bool:
        """`True` if the passed pathname is a valid pathname for the current OS;
        `False` otherwise."""
        try:
            if not isinstance(pathname, str) or not pathname:
                return False
            _, pathname = os.path.splitdrive(pathname)
            root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
                if sys.platform == 'win32' else os.path.sep
            assert os.path.isdir(root_dirname)
            root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep
            for pathname_part in pathname.split(os.path.sep):
                try:
                    os.lstat(root_dirname + pathname_part)
                except OSError as exc:
                    if hasattr(exc, 'winerror'):
                        # ERROR_INVALID_NAME = 123
                        if exc.winerror == 123:
                            return False
                    elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                        return False
        except TypeError as exc:
            return False
        else:
            return True

    def validate_image_file_name(self, file_name: str, detection_image_type: DetectionImageType) -> str:
        """validate image file name.
            Rule of detection image filename:
                Sample file name: 20200219_18802_16_11_1F7_ブロック1_r1
                1. after split by "_" then number of strings is 7
                2. the first part is datetime format
                3. the second part is body number. check is digit
            Rule of detection area image filename:
                Sample file name: E_20200219_18802_16_11_1F7_ブロック1_r1
                1. after split by "_" then number of strings is 8
                2. the second part is datetime format
                3. the third part is body number. check is digit
        """
        try:
            file_name_parts = file_name.split('_')
            if len(file_name_parts) != self.underscore_count_map[detection_image_type]:
                return self.image_format_incorrect_error_map[detection_image_type], 0

            datetime.datetime.strptime(file_name_parts[self.datetime_start_index_map[detection_image_type]], '%Y%m%d')

            if  not (len(file_name_parts[self.datetime_start_index_map[detection_image_type]]) == 8):
                return self.datetime_error_map[detection_image_type], 0

            if not file_name_parts[self.body_number_start_index_map[detection_image_type]].isdigit():
                return self.body_number_error_map[detection_image_type], 0

            body_number = int(file_name_parts[self.body_number_start_index_map[detection_image_type]])

            block_number_parts = file_name_parts[self.block_number_start_index_map[detection_image_type]].split("ブロック")
            if len(block_number_parts) != 2 or not block_number_parts[1].isdigit():
                return self.image_format_incorrect_error_map[detection_image_type], 0

            return ErrorCode.NONE, body_number
        except ValueError:
            return self.datetime_error_map[detection_image_type], 0
