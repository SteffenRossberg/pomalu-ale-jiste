import torch
from torch import nn as nn
from typing import Any
from src.networks.convolutionbase import ConvolutionBase


class Decoder(ConvolutionBase):

    def __init__(self, input_shape, output_shape):
        super(Decoder, self).__init__()
        kernel_size = (2, 1)
        self.output_shape = output_shape
        self.output_size1 = ConvolutionBase.calc_size(output_shape[0], kernel_size=kernel_size[0], stride=1, padding=0)
        self.output_size2 = ConvolutionBase.calc_size(self.output_size1, kernel_size=kernel_size[0], stride=1, padding=0)
        self.output_size3 = ConvolutionBase.calc_size(self.output_size2, kernel_size=kernel_size[0], stride=1, padding=0)
        self.decoder_hidden_layer1 = nn.Linear(in_features=input_shape, out_features=64)
        self.decoder_hidden_layer2 = nn.Linear(in_features=64, out_features=128 * self.output_size3 * output_shape[1])
        self.trans_convolution1 = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=1, padding=0)
        self.trans_convolution2 = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=1, padding=0)
        self.trans_convolution3 = nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, features):
        activation = self.decoder_hidden_layer1(features)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        activation = activation.view((features.size()[0], 128, self.output_size3, self.output_shape[1]))
        activation = self.trans_convolution1(activation)
        activation = torch.relu(activation)
        activation = self.trans_convolution2(activation)
        activation = torch.relu(activation)
        activation = self.trans_convolution3(activation)
        reconstructed = torch.relu(activation)
        reconstructed = activation.view((reconstructed.size()[0], 1, self.output_shape[0], self.output_shape[1]))
        return reconstructed

    def _forward_unimplemented(self, *features: Any) -> None:
        pass
