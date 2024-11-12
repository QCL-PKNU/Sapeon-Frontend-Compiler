from typing import Union, List

from abc import ABC, abstractmethod
from enum import Enum


class ActivationMode(Enum):
    Sigmoid = 0
    ReLU = 1
    LeakyReLU = 2
    PReLU = 3
    Tanh = 4
    Identity = 5
    Mish = 6
    ReLU6 = 7
    Swish = 8


class Activation(ABC):
    @abstractmethod
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.Identity


class Sigmoid(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.Sigmoid


class ReLU(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.ReLU


class LeakyReLU(Activation):
    def __init__(self, leaky_slope: float) -> None:
        self.leaky_slope: float = leaky_slope

    def activation_mode(self) -> ActivationMode:
        return ActivationMode.LeakyReLU


class PReLU(Activation):
    def __init__(self, neg_slope: Union[float, List[float]]) -> None:
        self.neg_slope: Union[float, List[float]] = neg_slope

    def activation_mode(self) -> ActivationMode:
        return ActivationMode.PReLU


class Tanh(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.Tanh


class Identity(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.Identity


class Mish(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.Mish


class ReLU6(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.ReLU6


class Swish(Activation):
    def activation_mode(self) -> ActivationMode:
        return ActivationMode.Swish
