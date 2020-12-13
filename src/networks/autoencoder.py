from typing import Any
from torch import nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, enc, dec):
        super(AutoEncoder, self).__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

    def _forward_unimplemented(self, *features: Any) -> None:
        pass
