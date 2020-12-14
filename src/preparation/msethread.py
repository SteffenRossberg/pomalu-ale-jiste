import threading
import numpy as np


class MseThread(threading.Thread):
    def __init__(self, x, y, offset_x, offset_y, match_threshold):
        threading.Thread.__init__(self)
        self.x = x
        self.y = y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.match_threshold = match_threshold
        self.indices = None

    def run(self):
        x = self.x
        y = np.swapaxes(self.y, 0, 1)
        diff = x - y
        square = diff * diff
        mse = np.mean(square, axis=(2, 3))
        self.indices = np.transpose(np.nonzero((mse < self.match_threshold))) + [self.offset_x, self.offset_y]
