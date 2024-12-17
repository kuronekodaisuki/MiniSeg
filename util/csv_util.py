# -*- coding: shift_jis -*-
import csv

from data.common_define import ErrorCode
class CsvUtil:
    """write csv result with utility class"""
    def __init__(self):
        pass

    @staticmethod
    def create_new_file(str_csv_filename):
        try:
            header_array1 = ['���ްNO','�ڰ�NO','���ް�敪','�Ԏ�NO','�O�F','��h��^�C�v','�c�[�g��','���[�����[�t','��h���','�ŏI��������','�s����','']
            header_array2 = ['�������t','��������','�ُ�R�[�h','�������u���b�N�L��','','','','','','','','']
            header_array3 = ['NO','�u���b�NNO','�����A�h���X','�����ʂ��A�h���X','���ׂw���W(mm)','���ׂx���W(mm)','����Z���W(mm)','���זʐ�(mm)','���׎��','�G���A�u���b�N','�ُ�R�[�h','�ŏ����a']
            row_content = ['', '', '', '', '', '', '', '', '', '', '', '']
            file_csv = open(str_csv_filename, mode="+a", encoding='shift-jis', newline = '')
            csv_writer = csv.writer(file_csv)
            csv_writer.writerow(header_array1)
            csv_writer.writerow(row_content)
            csv_writer.writerow(header_array2)
            csv_writer.writerow(row_content)
            csv_writer.writerow(header_array3)
            file_csv.close()
            return ErrorCode.NONE
        except Exception as e:
            return ErrorCode.CREATE_CSV_FAIL

    @staticmethod
    def write_data(file_path=None, data_to_write=None):
        try:
            no_index = data_to_write[0]
            detect_list = data_to_write[1]
            robo_no = data_to_write[2]
            file_csv = open(file_path, mode="+a", encoding='shift-jis', newline = '')
            csv_writer = csv.writer(file_csv)
            for detect_info in detect_list:
                row_content = ['', '', '', '', '', '', '', '', '', '', '', '']
                no_index = no_index + 1
                row_content[0] = format(no_index)
                row_content[1] = format(robo_no)
                row_content[4] = format(round((detect_info[0] + detect_info[2]) / 2))
                row_content[5] = format(round((detect_info[1] + detect_info[3]) / 2))
                row_content[11] = format(max(abs(detect_info[2] - detect_info[0]), abs(detect_info[3] - detect_info[1])))
                csv_writer.writerow(row_content)
            file_csv.close()
            return ErrorCode.NONE
        except Exception as e:
            return ErrorCode.WRITE_DATA_CSV_FAIL
