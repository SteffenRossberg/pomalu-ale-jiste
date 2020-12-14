from typing import Any
from torch import nn as nn


class ConvolutionBase(nn.Module):

    def __init__(self):
        super(ConvolutionBase, self).__init__()

    @staticmethod
    def calc_size(input_size, kernel_size, stride, padding):
        return int(((input_size - kernel_size + (2 * padding)) / stride) + 1)

    def _forward_unimplemented(self, *features: Any) -> None:
        pass
