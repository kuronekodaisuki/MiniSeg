class BlockInfo(object):
    """Structure of a block information"""
    def __init__(self, body_number, total_blocks, block_number, detection_image_file_name,
                 detection_area_image_file_name):
        self.body_number = body_number
        self.total_blocks = total_blocks
        self.block_number = block_number
        self.detection_image_file_name = detection_image_file_name
        self.detection_area_image_file_name = detection_area_image_file_name
