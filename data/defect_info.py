class DefectInfo:
    """Structure of car panel inspection result"""
    def __init__(self):
        """
            Defect info structure
        """
        self.defect_infos = list()    # len(defect_infos) is the number of defects
                                      # [xmin, ymin, xmax, ymax] correspounding to original image