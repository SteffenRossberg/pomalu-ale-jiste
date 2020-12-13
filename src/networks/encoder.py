import torch
from torch import nn as nn
from typing import Any
from src.networks.convolutionbase import ConvolutionBase


class Encoder(ConvolutionBase):

    def __init__(self, input_shape, output_shape):
        super(Encoder, self).__init__()
        kernel_size = (2, 1)
        output_size1 = ConvolutionBase.calc_size(input_shape[0], kernel_size=kernel_size[0], stride=1, padding=0)
        output_size2 = ConvolutionBase.calc_size(output_size1, kernel_size=kernel_size[0], stride=1, padding=0)
        output_size3 = ConvolutionBase.calc_size(output_size2, kernel_size=kernel_size[0], stride=1, padding=0)
        self.convolution1 = nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=0)
        self.convolution2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=0)
        self.convolution3 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=0)
        self.encoder_hidden_layer1 = nn.Linear(in_features=128 * output_size3 * input_shape[1], out_features=64)
        self.encoder_output_layer = nn.Linear(in_features=64, out_features=output_shape)

    def forward(self, features):
        activation = self.convolution1(features)
        activation = torch.relu(activation)
        activation = self.convolution2(activation)
        activation = torch.relu(activation)
        activation = self.convolution3(activation)
        activation = torch.relu(activation)
        activation = activation.view(features.size()[0], -1)
        activation = self.encoder_hidden_layer1(activation)
        activation = torch.relu(activation)
        encoded = self.encoder_output_layer(activation)
        encoded = torch.relu(encoded)
        return encoded

    def _forward_unimplemented(self, *features: Any) -> None:
        pass

