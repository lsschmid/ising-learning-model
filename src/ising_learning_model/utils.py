"""Utility functions for the Ising learning model."""
from __future__ import annotations
import numpy as np


class GammaInitialization:
    """Gamma initialization settings for the model."""

    mode: str
    value_range: None | tuple[float, float]
    _gamma_init: np.array

    def __init__(self, mode: str) -> None:
        if mode in ["zeros", "random", "fixed"]:
            self.mode = mode
            self.value_range = None
            self._gamma_init = None
        else:
            msg = "invalid gamma initialization mode, choose zeros or random"
            raise ValueError(msg)

    def initialize(
        self, size: int, value_range: tuple[float, float] = None
    ) -> np.array:
        """Initialize the gamma values."""
        gamma = np.zeros((size, size))
        if self.mode == "zeros":
            return gamma
        elif self.mode == "random":
            if value_range is None and self.value_range is None:
                msg = (
                    "value range for random gamma initialization not specified"
                )
                raise ValueError(msg)
            if self.value_range is None:
                self.value_range = value_range
            for i in range(size):
                for j in range(i + 1, size):
                    gamma[i, j] = np.random.uniform(
                        self.value_range[0], self.value_range[1]
                    )
            return gamma
        elif self.mode == "fixed":
            if self._gamma_init is None:
                msg = "fixed gamma initialization not specified"
                raise ValueError(msg)
            if self._gamma_init.shape != gamma.shape:
                msg = "matrix of fixed gamma initialization has wrong shape"
                raise ValueError(msg)
            return self._gamma_init


class utils:
    @staticmethod
    def mse_loss_function(y_prediction: float, y_true: float) -> float:
        """MSE loss function."""
        return float((y_prediction - y_true) ** 2)

    @staticmethod
    def vector_to_biases(theta: np.array) -> dict:
        """
        Convert the theta vector to biases of an Ising model.

        param theta: the theta vector
        type: np.array

        return: the bias values
        rtype: dict
        """
        return {k: v for k, v in enumerate(theta.tolist())}

    @staticmethod
    def gamma_to_couplings(gamma: np.array) -> dict:
        """
        Convert the gamma matrix to couplings of an Ising model.

        param gamma: the gamma matrix
        type: np.array

        return: the coupling values
        rtype: dict
        """
        J = {
            (qubit_i, qubit_j): weight
            for (qubit_i, qubit_j), weight in np.ndenumerate(gamma)
            if qubit_i < qubit_j
        }
        return J

    @staticmethod
    def make_upper_triangular(gamma: np.array) -> np.array:
        """
        Convert the given gamma matrix to an upper triangular matrix.

        param gamma: the gamma matrix
        type: np.array

        return: the upper triangular gamma matrix
        rtype: np.array
        """
        size = gamma.shape[0]
        for i in range(size):
            gamma[i, i] = 0
            for j in range(i + 1, size):
                gamma[j, i] = 0
        return gamma
