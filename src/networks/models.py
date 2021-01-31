import torch
from torch import nn as nn
from typing import Any
from collections import OrderedDict


class Trader(nn.Module):

    def __init__(self, days, state_size=7):
        super(Trader, self).__init__()
        self.days = days
        self.state_size = state_size
        self.buyer = self._create_auto_encoder()
        self.seller = self._create_auto_encoder()
        self.decision_maker = self._create_decision_maker()

    def forward(self, features):
        encoder_features = features[:, :features.shape[1] - self.state_size]
        encoder_features = encoder_features.view((features.shape[0], 1, self.days, 4))
        encoder_features = encoder_features.to(features.device)

        buyer_prediction = self.buyer(encoder_features)
        seller_prediction = self.seller(encoder_features)

        buyer_mse = self.__calculate_mse(buyer_prediction, encoder_features)
        seller_mse = self.__calculate_mse(seller_prediction, encoder_features)
        detector_features = self.__merge_mse(buyer_mse, seller_mse)

        state_features = features[:, -self.state_size:]
        state_features = state_features.view((features.shape[0], 1, 1, self.state_size))
        state_features = state_features.to(features.device)
        decision_features = self.__merge_state(detector_features, state_features)

        decision = self.decision_maker(decision_features)
        return decision

    def _forward_unimplemented(self, *features: Any) -> None:
        pass

    def _create_auto_encoder(self):
        encoder = Encoder(input_shape=(self.days, 4), output_size=self.days)
        decoder = Decoder(input_size=self.days, output_shape=(self.days, 4))
        auto_encoder = AutoEncoder(encoder, decoder)
        return auto_encoder

    def _create_decision_maker(self):
        selector = DecisionMaker(self.state_size)
        return selector

    def reset_buyer(self, device):
        self.buyer = self._create_auto_encoder().to(device)

    def reset_seller(self, device):
        self.seller = self._create_auto_encoder().to(device)

    @staticmethod
    def __calculate_mse(x, y):
        diff = x - y
        diff = torch.squeeze(diff, dim=0)
        square = diff * diff
        square = square.view((square.shape[0], 1, square.shape[-2], square.shape[-1]))
        mse = torch.mean(square, dim=(2, 3))
        return mse

    @staticmethod
    def __merge_mse(buyer_mse, seller_mse):
        seller_shape = seller_mse.shape
        seller_reshaped = seller_mse.view((seller_shape[1], seller_shape[0]))
        merged_mse = torch.cat((buyer_mse, torch.t(seller_reshaped)), 1)
        return merged_mse

    def __merge_state(self, features, state):
        prepared_features = self.__prepare_data(features)
        prepared_state = self.__prepare_data(state)
        merged_features = torch.cat((prepared_features, prepared_state), 1)
        result = merged_features.view((features.shape[0], 1, 1, merged_features.shape[-1]))
        return result

    @staticmethod
    def __prepare_data(data):
        data_shape = data.shape
        prepared_data = data.view((data_shape[0], data_shape[-1]))
        return prepared_data


class DecisionMaker(nn.Module):
    def __init__(self, state_size):
        super(DecisionMaker, self).__init__()
        self.state_size = state_size
        in_features = 2 + state_size
        self.adv = nn.Sequential(
            NetHelper.init_weights(
                nn.Linear(
                    in_features=in_features,
                    out_features=512)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=512,
                    out_features=512)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=512,
                    out_features=3)),
        )
        self.val = nn.Sequential(
            NetHelper.init_weights(
                nn.Linear(
                    in_features=in_features,
                    out_features=512)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=512,
                    out_features=512)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=512,
                    out_features=1)),
        )

    def forward(self, features):
        val = self.val(features)
        adv = self.adv(features)
        observation = val + adv - adv.mean(dim=3, keepdim=True)
        observation = observation.view((features.shape[0], 3))
        return observation

    def _forward_unimplemented(self, *features: Any) -> None:
        pass


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
            layers[key] = NetHelper.init_weights(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel,
                    stride,
                    padding))
            self.shape = ConvolutionHelper.calc_2d_size(self.shape, kernel, stride, padding)
            layers[key + '_pool'] = nn.MaxPool2d(kernel_size=kernel, padding=(1, 0), stride=(1, 1))
            self.shape = ConvolutionHelper.calc_2d_size(self.shape, kernel, padding=(1, 0), stride=(1, 1))
            layers[key + '_drop_out'] = nn.Dropout2d(0.2)
            layers[key + '_activation'] = nn.LeakyReLU()
            self.in_channels = self.out_channels
            self.out_channels += int(self.out_channels / 4)
        self.out_channels = self.in_channels
        self.conv_out_size = self.out_channels * self.shape[0] * self.shape[1]

        self.convolution = nn.Sequential(layers)
        self.linear = nn.Sequential(
            NetHelper.init_weights(
                nn.Linear(
                    in_features=self.conv_out_size,
                    out_features=int(self.conv_out_size / 4))),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=int(self.conv_out_size / 4),
                    out_features=output_size)),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )

    def forward(self, features):
        result = self.convolution(features)
        result = result.view((features.size()[0], 1, 1, result.shape[-3] * result.shape[-2] * result.shape[-1]))
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
            layers[key] = NetHelper.init_weights(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel,
                    stride,
                    padding))
            self.shape = ConvolutionHelper.calc_2d_size(self.shape, kernel, stride, padding)
            layers[key + '_drop_out'] = nn.Dropout2d(0.2)
            layers[key + '_activation'] = nn.LeakyReLU()

        self.transpose = nn.Sequential(layers)
        self.out_channels = channels[-1]

        self.linear = nn.Sequential(
            NetHelper.init_weights(
                nn.Linear(
                    in_features=self.input_size,
                    out_features=128)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=128,
                    out_features=self.out_channels * self.shape[0] * self.shape[1])),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )

    def forward(self, features):
        result = self.linear(features)
        result = result.view((features.size()[0], self.out_channels, self.shape[0], self.shape[1]))
        result = self.transpose(result)
        return result

    def _forward_unimplemented(self, *features: Any) -> None:
        pass


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


class NetHelper:
    @staticmethod
    def init_weights(layer, mean=0.0, std=0.05):
        torch.nn.init.normal_(layer.weight, mean=mean, std=std)
        return layer
