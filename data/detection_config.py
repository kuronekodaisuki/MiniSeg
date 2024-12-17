class DetectionConfig(object):
    """Structure of detection config"""
    def __init__(self, confidence_threshold, resized_height, resized_width, draw_result, detection_size, debug = 0):
        self.confidence_threshold = confidence_threshold
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.draw_result = draw_result
        self.detection_size = detection_size
        self.debug = debug