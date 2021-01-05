from torch import nn as nn
from typing import Any
from collections import OrderedDict


class ConvolutionHelper:

    @staticmethod
    def calc_2d_size(shape, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        height = ConvolutionHelper.calc_1d_size(shape[0], kernel[0], stride[0], padding[0], dilation[0])
        width = ConvolutionHelper.calc_1d_size(shape[1], kernel[1], stride[1], padding[1], dilation[1])
        return height, width

    @staticmethod
    def calc_1d_size(size, kernel, stride=1, padding=0, dilation=1):
        padding *= 2
        kernel = dilation * (kernel - 1)
        return int(((size + padding - kernel - 1) / stride) + 1)


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

    def _forward_unimplemented(self, *features: Any) -> None:
        pass


class Encoder(nn.Module):

    def __init__(self, input_shape, output_size, conv_layer_count=4):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        kernel = (2, 1)
        self.shape = input_shape
        self.in_channels = 1
        self.out_channels = 32
        stride = (1, 1)
        padding = (0, 0)
        layers = OrderedDict()
        for i in range(conv_layer_count):
            key = f'layer{i}'
            layers[key] = nn.Conv2d(self.in_channels, self.out_channels, kernel, stride, padding)
            self.shape = ConvolutionHelper.calc_2d_size(self.shape, kernel, stride, padding)
            layers[key + '_pool'] = nn.MaxPool2d(kernel_size=kernel, padding=(1, 0), stride=(1, 1))
            self.shape = ConvolutionHelper.calc_2d_size(self.shape, kernel, padding=(1, 0), stride=(1, 1))
            layers[key + '_drop_out'] = nn.Dropout2d(0.2)
            layers[key + '_activation'] = nn.ReLU()
            self.in_channels = self.out_channels
            self.out_channels += int(self.out_channels / 4)
        self.out_channels = self.in_channels
        self.conv_out_size = self.out_channels * self.shape[0] * self.shape[1]

        self.convolution = nn.Sequential(layers)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.conv_out_size, out_features=int(self.conv_out_size / 4)),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(in_features=int(self.conv_out_size / 4), out_features=output_size),
            nn.Dropout(0.2),
            nn.Tanh()
        )

    def forward(self, features):
        result = self.convolution(features)
        result = result.reshape(features.size()[0], 1, 1, result.shape[-3] * result.shape[-2] * result.shape[-1])
        result = self.linear(result)
        return result

    def _forward_unimplemented(self, *features: Any) -> None:
        pass


class Decoder(nn.Module):

    def __init__(self, input_size, output_shape, conv_layer_count=4):
        super(Decoder, self).__init__()

        self.input_size = input_size
        kernel = (2, 1)
        stride = (1, 1)
        padding = (1, 0)

        self.shape = output_shape
        self.in_channels = 32

        channels = [1, self.in_channels]
        for i in range(conv_layer_count - 1):
            channels.append(int(channels[-1] * 1.25))

        layers = OrderedDict()
        for i in range(len(channels) - 1, 0, -1):
            key = f'layer{i - 1}'
            in_channel = channels[i]
            out_channel = channels[i - 1]
            layers[key] = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding)
            self.shape = ConvolutionHelper.calc_2d_size(self.shape, kernel, stride, padding)
            layers[key + '_drop_out'] = nn.Dropout2d(0.2)
            layers[key + '_activation'] = nn.ReLU()

        self.transpose = nn.Sequential(layers)
        self.out_channels = channels[-1]

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=128),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=self.out_channels * self.shape[0] * self.shape[1]),
            nn.Dropout(0.2),
            nn.Tanh()
        )

    def forward(self, features):
        result = self.linear(features)
        result = result.reshape(features.size()[0], self.out_channels, self.shape[0], self.shape[1])
        result = self.transpose(result)
        return result

    def _forward_unimplemented(self, *features: Any) -> None:
        pass
