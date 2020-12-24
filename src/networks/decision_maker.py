from typing import Any
import torch
from torch import nn as nn


class DecisionMaker(nn.Module):

    def __init__(self, classifier, state_size=6, action_size=3):
        super(DecisionMaker, self).__init__()
        self.classifier = classifier
        in_features = state_size + action_size
        self.decision_maker = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features * 32),
            nn.ReLU(),
            nn.Linear(in_features=in_features * 32, out_features=in_features * 64),
            nn.ReLU(),
            nn.Linear(in_features=in_features * 64, out_features=action_size),
            nn.Tanh()
        )

    def forward(self, features, **kwargs):
        classification = self.classifier(features)
        state = torch.tensor(kwargs['state'], dtype=classification.dtype).to(classification.device)
        merged_classification = self.__merge_state(classification, state)
        observation = self.decision_maker(merged_classification)
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
