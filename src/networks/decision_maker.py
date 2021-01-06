from typing import Any
import torch
from torch import nn as nn


class DecisionMaker(nn.Module):

    def __init__(self, classifier, state_size=3, days=5):
        super(DecisionMaker, self).__init__()
        self.state_size = state_size
        self.days = days
        self.classifier = classifier
        self.selector = Selector(self.state_size)

    def forward(self, features):
        classification_features = features[:, :features.shape[1] - self.state_size]
        classification_features = classification_features.reshape(features.shape[0], 1, self.days, 4)
        classification_features = classification_features.to(features.device)

        state_features = features[:, -self.state_size:]
        state_features = state_features.reshape(features.shape[0], 1, 1, self.state_size)
        state_features = state_features.to(features.device)

        classification = self.classifier(classification_features)
        merged_classification = self.__merge_state(classification, state_features)
        observation = self.selector(merged_classification)
        return observation

    def _forward_unimplemented(self, *features: Any) -> None:
        pass

    @staticmethod
    def __merge_state(features, state):
        prepared_features = DecisionMaker.__prepare_data(features)
        prepared_state = DecisionMaker.__prepare_data(state)
        merged_features = torch.cat((prepared_features, prepared_state), 1)
        result = merged_features.reshape(features.shape[0], 1, 1, merged_features.shape[-1])
        return result

    @staticmethod
    def __prepare_data(data):
        data_shape = data.shape
        prepared_data = data.reshape(data_shape[0], data_shape[-1])
        return prepared_data


class Selector(nn.Module):
    def __init__(self, state_size):
        super(Selector, self).__init__()
        self.state_size = state_size
        action_size = 3  # none = 0, buy = 1, sell = 2
        in_features = action_size + state_size
        self.adv = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=512),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=3),
        )
        self.val = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=512),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, features):
        val = self.val(features)
        adv = self.adv(features)
        observation = val + adv - adv.mean(dim=3, keepdim=True)
        observation = observation.reshape(features.shape[0], 3)
        return observation

    def _forward_unimplemented(self, *features: Any) -> None:
        pass
