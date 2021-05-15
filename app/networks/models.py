import torch
from torch import nn as nn
from typing import Any
from collections import OrderedDict
import pandas as pd


class Trader(nn.Module):

    def __init__(self, days, state_size=7):
        super(Trader, self).__init__()
        self.days = days
        self.state_size = state_size
        self.buyer = self._create_auto_encoder()
        self.seller = self._create_auto_encoder()
        self.classifier = self._create_classifier()

    def forward(self, features):
        encoder_features = features[:, :features.shape[1] - self.state_size]
        encoder_features = encoder_features.view((features.shape[0], 1, 4, self.days))
        encoder_features = encoder_features.to(features.device)

        # run auto encoders
        classifier_features = self.forward_encoders(encoder_features)

        # run classifier
        classifier_prediction = self.classifier(classifier_features)

        decision = classifier_prediction.view((classifier_prediction.shape[0], 3))
        return decision

    def _forward_unimplemented(self, *features: Any) -> None:
        pass

    def forward_encoders(self, features):
        buyer_prediction = self.buyer(features)
        seller_prediction = self.seller(features)
        buyer_mse = self.__calculate_ratio(buyer_prediction, features)
        seller_mse = self.__calculate_ratio(seller_prediction, features)
        prediction = self.__merge_ratio(buyer_mse, seller_mse)
        return prediction

    def _create_auto_encoder(self):
        encoder = Encoder(input_shape=(4, self.days), output_size=self.days)
        decoder = Decoder(input_size=self.days, output_shape=(4, self.days))
        auto_encoder = AutoEncoder(encoder, decoder)
        return auto_encoder

    @staticmethod
    def _create_classifier():
        classifier = Classifier()
        return classifier

    def reset_buyer(self, device):
        self.buyer = self._create_auto_encoder().to(device)

    def reset_seller(self, device):
        self.seller = self._create_auto_encoder().to(device)

    def reset_classifier(self, device):
        self.classifier = self._create_classifier().to(device)

    @staticmethod
    def __calculate_ratio(x, y):
        diff = x - y
        diff = torch.squeeze(diff, dim=0)
        square = diff * diff
        square = square.view((square.shape[0], 1, square.shape[-2], square.shape[-1]))
        mse = torch.mean(square, dim=(2, 3))
        return mse

    @staticmethod
    def __merge_ratio(buyer_mse, seller_mse):
        seller_shape = seller_mse.shape
        seller_reshaped = seller_mse.view((seller_shape[1], seller_shape[0]))
        merged_mse = torch.cat((buyer_mse, torch.t(seller_reshaped)), 1)
        return merged_mse


class TrainClassifier(nn.Module):

    def __init__(self, trader):
        super(TrainClassifier, self).__init__()
        self.trader = trader

    def forward(self, features):
        # run auto encoders
        classifier_features = self.trader.forward_encoders(features)
        # run classifier
        prediction = self.trader.classifier(classifier_features)
        return prediction

    def _forward_unimplemented(self, *features: Any) -> None:
        pass


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            NetHelper.init_weights(nn.Linear(2, 256)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(nn.Linear(256, 256)),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            NetHelper.init_weights(nn.Linear(256, 3)),
            nn.Sigmoid()
        )

    def forward(self, features):
        prediction = self.classifier(features)
        return prediction

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
            layers[key + '_activation'] = nn.LeakyReLU()
            self.in_channels = self.out_channels
            self.out_channels += int(self.out_channels / 4)
        self.out_channels = self.in_channels
        self.conv_out_size = self.out_channels * self.shape[0] * self.shape[1]

        self.convolution = nn.Sequential(layers)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=self.conv_out_size,
                    out_features=int(self.conv_out_size / 4))),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=int(self.conv_out_size / 4),
                    out_features=output_size)),
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
            layers[key + '_activation'] = nn.LeakyReLU()

        self.transpose = nn.Sequential(layers)
        self.out_channels = channels[-1]
        self.conv_out_size = self.out_channels * self.shape[0] * self.shape[1]

        self.linear = nn.Sequential(
            NetHelper.init_weights(
                nn.Linear(
                    in_features=self.input_size,
                    out_features=int(self.conv_out_size / 4))),
            nn.LeakyReLU(),
            NetHelper.init_weights(
                nn.Linear(
                    in_features=int(self.conv_out_size / 4),
                    out_features=self.out_channels * self.shape[0] * self.shape[1])),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, features):
        result = self.linear(features)
        result = result.view((features.size()[0], self.out_channels, self.shape[0], self.shape[1]))
        result = self.transpose(result)
        return result

    def _forward_unimplemented(self, *features: Any) -> None:
        pass


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = [1] + [dimension for dimension in shape]

    def forward(self, x):
        self.shape[0] = x.shape[0]
        return x.view(*self.shape)


class TrainableModel(nn.Module):
    def __init__(self, loss_function):
        super(TrainableModel, self).__init__()
        self.loss_function = loss_function
        self.optimizer = None
        self.counter = 0
        self.progress = []

    def _train_step(self, outputs, targets):
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=0, figsize=(16, 8), alpha=0.3, marker='.', grid=True, yticks=(0, 0.025, 0.05, 0.1, 0.2))


class AutoEncoder2(TrainableModel):
    def __init__(self):
        super(AutoEncoder2, self).__init__(nn.MSELoss())
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(),

            nn.Conv2d(30, 50, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),

            nn.Conv2d(50, 50, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.LeakyReLU(),

            View([50 * 1 * 2]),
            nn.Linear(50 * 1 * 2, 10),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 50 * 1 * 2),
            nn.LeakyReLU(),

            View([50, 1, 2]),

            nn.ConvTranspose2d(50, 50, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(50, 30, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(30, 1, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(1),

            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def train_net(self, inputs, targets):
        outputs = self(inputs)
        self._train_step(outputs, targets)


class Classifier2(TrainableModel):

    def __init__(self):
        super(Classifier2, self).__init__(nn.CrossEntropyLoss())
        self._classifier = nn.Sequential(
            nn.Conv2d(2, 30, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(),

            nn.Conv2d(30, 60, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(60),
            nn.LeakyReLU(),

            nn.Conv2d(60, 60, kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.LeakyReLU(),

            View([60 * 2 * 1]),
            nn.Linear(60 * 2 * 1, 3),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs):
        return self._classifier(inputs)

    def train_net(self, auto_encoder, inputs, targets):
        decoded = auto_encoder(inputs)
        features = torch.cat([inputs, decoded], dim=1)
        outputs = self(features.detach())
        self._train_step(outputs, targets)


class Trader2(nn.Module):

    def __init__(self, auto_encoder: AutoEncoder2, classifier: Classifier2):
        super(Trader2, self).__init__()
        self._auto_encoder = auto_encoder
        self._classifier = classifier
        self.state_size = 2
        self.days = 5

    def forward(self, inputs):
        encoder_features = inputs[:, :inputs.shape[1] - self.state_size]
        encoder_features = encoder_features.view((inputs.shape[0], 1, 4, self.days))
        inputs = encoder_features.to(inputs.device)

        decoded = self._auto_encoder(inputs)
        features = torch.cat([inputs, decoded], dim=1)
        outputs = self._classifier(features.detach())
        return outputs


class ConvolutionHelper:

    @classmethod
    def calc_2d_size(cls, shape, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        return cls._calculate(shape, kernel, stride, padding, dilation, callback=cls.calc_1d_size)

    @classmethod
    def calc_2d_transpose_size(cls, shape, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        return cls._calculate(shape, kernel, stride, padding, dilation, callback=cls.calc_1d_transpose_size)

    @classmethod
    def _calculate(cls, shape, kernel, stride, padding, dilation, callback):
        height = callback(shape[0], kernel[0], stride[0], padding[0], dilation[0])
        width = callback(shape[1], kernel[1], stride[1], padding[1], dilation[1])
        return height, width

    @staticmethod
    def calc_1d_size(size, kernel, stride=1, padding=0, dilation=1):
        padding *= 2
        kernel = dilation * (kernel - 1)
        return int(((size + padding - kernel - 1) / stride) + 1)

    @staticmethod
    def calc_1d_transpose_size(size, kernel, stride=1, padding=0, dilation=1):
        padding *= 2
        kernel = dilation * (kernel - 1)
        return int(((size - 1) * stride) + 1 + kernel - padding)


class NetHelper:
    @staticmethod
    def init_weights(layer, mean=0.0, std=0.05):
        torch.nn.init.normal_(layer.weight, mean=mean, std=std)
        return layer
