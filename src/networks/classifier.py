from typing import Any
import torch
from torch import nn as nn


class Classifier(nn.Module):

    def __init__(self, buyer_auto_encoder, seller_auto_encoder):
        super(Classifier, self).__init__()
        self.buyer_auto_encoder = buyer_auto_encoder
        self.seller_auto_encoder = seller_auto_encoder
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=3),
            nn.Tanh()
        )

    def forward(self, features):
        buyer_prediction = self.buyer_auto_encoder(features)
        seller_prediction = self.seller_auto_encoder(features)
        buyer_mse = self.__calculate_mse(buyer_prediction, features)
        seller_mse = self.__calculate_mse(seller_prediction, features)
        merged = self.__merge_mse(buyer_mse, seller_mse)
        classification = self.classifier(merged)
        return classification

    def _forward_unimplemented(self, *features: Any) -> None:
        pass

    @staticmethod
    def __calculate_mse(x, y):
        diff = x - y
        diff = torch.squeeze(diff, dim=0)
        square = diff * diff
        mse = torch.mean(square, dim=(2, 3))
        return mse

    @staticmethod
    def __merge_mse(buyer_mse, seller_mse):
        seller_shape = seller_mse.shape
        seller_reshaped = seller_mse.reshape(seller_shape[1], seller_shape[0])
        merged = torch.cat((buyer_mse, torch.t(seller_reshaped)), 1)
        result = merged.reshape(merged.shape[0], merged.shape[1])
        return result
