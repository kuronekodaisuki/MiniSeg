class BlockProcessResult(object):
    """Structure of the process result for a block"""
    def __init__(self, block_number):
        self.block_number = block_number
        self.is_completed = False
