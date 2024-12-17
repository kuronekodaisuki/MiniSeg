import os
import shutil
from data.directory_config import DirectoryConfig


class DirectoryUtil:
    """working with directory, file with utility class"""
    def __init__(self):
        self.directory_config = DirectoryConfig()
        pass

    def initialize(self, directory_config):
        self.directory_config.img_source = directory_config.img_source
        self.directory_config.img_dest = directory_config.img_dest
        self.directory_config.csv_dest = directory_config.csv_dest
        self.directory_config.log_dest = directory_config.log_dest

    def make_img_result_folder(self, folder_path):
        """make folder with safe method"""
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            return True
        except Exception as e:
            return False

    def delete_file(self, file_name):
        """delete file with safe method"""
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
            return True
        except Exception as e:
            return False

    def move_file(self, file_name, dest_folder):
        """move file with safe method"""
        try:
            if os.path.exists(file_name) and os.path.exists(dest_folder):
                str_src = os.path.join(self.directory_config.img_source, file_name)
                str_dest = os.path.join(dest_folder, file_name)
                shutil.move(str_src, str_dest)
                return True
            return False
        except Exception as e:
            return False
