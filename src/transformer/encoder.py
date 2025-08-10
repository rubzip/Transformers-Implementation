from abc import ABC, abstractmethod

from torch import nn


class Encoder(nn.Module):


    @abstractmethod
    def forward(self, input_embedding, positional_embedding):
        pass


