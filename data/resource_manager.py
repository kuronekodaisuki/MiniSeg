# -*- coding: shift_jis -*-
from data.common_define import ErrorCode

class ResourceManager:
    """Structure of a block information"""

    def __init__(self):
        self.resource_dict = {
            ErrorCode.LOAD_CONFIG_FAIL: "configs�t�@�C���̃��[�h�Ɏ��s���܂����B",
            ErrorCode.DETECTION_CONFIG_INVALID: "�����p�����[�^���������ݒ肳��Ă��܂���B",
            ErrorCode.INPUT_IMAGE_FOLDER_PATH_INCORRECT: "�摜�̓��͕ۑ��t�H���_�[���������ݒ肳��Ă��܂���B",
            ErrorCode.OUTPUT_IMAGE_FOLDER_PATH_INCORRECT: "�摜�̏o�͕ۑ��t�H���_�[���������ݒ肳��Ă��܂���B",
            ErrorCode.CSV_FOLDER_PATH_INCORRECT: "csv�ۑ��t�H���_�[���������ݒ肳��Ă��܂���B",
            ErrorCode.LOG_FOLDER_PATH_INCORRECT: "���O�ۑ��t�H���_�[���������ݒ肳��Ă��܂���B",
            ErrorCode.MODEL_NOT_EXIST: "model�t�@�C����������܂���B",
            ErrorCode.LOAD_MODEL_FAIL: "",
            ErrorCode.READ_LINE_INFO_FAIL: "�u%s�v�̓ǂݍ��݂Ɏ��s���܂����B",
            ErrorCode.LINE_INFO_TEXT_NAME_INCORRECT: "�u%s�v�̃t�@�C����������������܂���B",
            ErrorCode.TOTAL_BLOCK_INCORRECT: "�u%s�v�̃u���b�N�������������͂���Ă��܂���B",
            ErrorCode.DETECTION_IMAGE_FORMAT_INCORRECT: "�u%s�v�����摜�̃t�@�C����������������܂���B(_�s��)",
            ErrorCode.DETECTION_IMAGE_DATETIME_INCORRECT: "�u%s�v�����摜�̃t�@�C����������������܂���B(���t�s��)",
            ErrorCode.DETECTION_IMAGE_BODY_NUMBER_INCORRECT: "�u%s�v�����摜�̃t�@�C����������������܂���B(�{�f�B�ԍ�)",
            ErrorCode.DETECTION_AREA_IMAGE_FORMAT_INCORRECT: "�u%s�v�����̈�摜�̃t�@�C����������������܂���B(_�s��)",
            ErrorCode.DETECTION_AREA_IMAGE_DATETIME_INCORRECT: "�u%s�v�����̈�摜�̃t�@�C����������������܂���B(���t�s��)",
            ErrorCode.DETECTION_AREA_IMAGE_BODY_NUMBER_INCORRECT: "�u%s�v�����̈�摜�̃t�@�C����������������܂���B(�{�f�B�ԍ�)",
            ErrorCode.DETECTION_IMAGE_NOT_FOUND: "�u%s�v�����摜�����݂��܂���B",
            ErrorCode.CANNOT_READ_IMAGE: "�u%s�v�̉摜�̓ǂݍ��݂Ɏ��s���܂����B",
            ErrorCode.SPLIT_IMAGE_FAIL: "�u%s�v�̉摜�̕����Ɏ��s���܂����B",
            ErrorCode.MASKING_IMAGE_FAIL: "�u%s�v�̃}�X�L���O�Ɏ��s���܂����B",
            ErrorCode.DRAW_BOUND_FAIL: "�u%s�v�̃o�E���f�B���O�{�b�N�X�̕`��Ɏ��s���܂����B",
            ErrorCode.COMBINE_RESULT_IMAGE_FAIL: "�u%s�v�̌������ʉ摜�����Ɏ��s���܂����B",
            ErrorCode.DETECTION_NO_MASKING_FAIL: "�u%s�v�̃}�X�N�����̐��_�Ɏ��s���܂����B",
            ErrorCode.DETECTION_MASKING_FAIL: "�u%s�v�̃}�X�N�L��̐��_�Ɏ��s���܂����B",
            ErrorCode.CREATE_OUTPUT_IMAGE_FOLDER_FAIL: "�u%s�v�������ʉ摜�̕ۑ��t�H���_�쐬�Ɏ��s���܂����B",
            ErrorCode.WRITE_RESULT_IMAGE_FAIL: "�u%s�v�������ʉ摜�̏������݂Ɏ��s���܂����B",
            ErrorCode.CREATE_CSV_FAIL: "�u%s�v�������t�@�C��(�ꎞ)�̍쐬�Ɏ��s���܂����B",
            ErrorCode.WRITE_DATA_CSV_FAIL: "�u%s�v�������t�@�C���̏������݂Ɏ��s���܂����B",
            ErrorCode.MOVE_CSV_FAIL: "�u%s�v�������t�@�C���̈ړ��Ɏ��s���܂����B",
            ErrorCode.DELETE_IMAGE_FAIL: "�u%s�v�����Ώۃt�@�C���̍폜�Ɏ��s���܂����B",
            ErrorCode.WRITE_LOG_FAIL: "�u%s�vlog�t�@�C���̏������݂Ɏ��s���܂����B",
            ErrorCode.FILE_EXTENSION_INCORRECT: "�u%s�v�t�@�C���g���q������������܂���B",
            ErrorCode.FILE_EXTENSION_INCORRECT_NOT_TIF: "�u%s�v�u%s�v�t�@�C���g���q������������܂���B",
            ErrorCode.FILE_SIZE_ERROR: "�u%s�v�u%s�v�t�@�C���e�ʂ��K��𒴂��Ă��܂��B",
            ErrorCode.DETECTION_CONFIG_SIZE_ERROR: "�����T�C�Y���������ݒ肳��Ă��܂���B",
            ErrorCode.DETECTION_CONFIG_SIZE_INVALID_VALUE: "�����T�C�Y臒l���w��\�͈͊O�ł��B",
        }

    def get_string(self, error_code):
        if self.resource_dict.get(error_code) is not None:
            return self.resource_dict.get(error_code)
        return ""
