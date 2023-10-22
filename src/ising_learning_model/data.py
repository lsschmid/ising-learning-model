""" Dataset class for ising learning model,
which also return the index for mini batches. """
from __future__ import annotations
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

from ising_learning_model.utils import GammaInitialization, utils
from dimod import ExactSolver


class HiddenNodesInitialization:
    """Hidden nodes initialization settings for the model."""

    mode: str = "repeat"
    random_range: tuple[torch.Tensor, torch.Tensor] = (-1, 1)
    function: callable = None
    fun_args: tuple = None

    def __init__(self, mode) -> None:
        self._function = None
        if mode == "repeat" or "random" or "zeros":
            self.mode = mode
        elif mode == "function":
            self.mode = mode
            self._function = lambda theta, index_new: theta[
                index_new % len(theta)
            ]
        else:
            msg = "invalid gamma initialization mode"
            raise ValueError(msg)



class SimpleDataset(Dataset):
    """ Dataset class for ising learning model """
    x = torch.Tensor
    y = torch.Tensor
    len = int
    data_size = int
    _gamma_data = np.array
    _ising_configs = list

    def __init__(self):
        super().__init__()

    def create_data_ising(
        self,
        size: int,
        num_samples: int,
        value_range_biases: tuple[float, float] = (-1, 1),
        value_range_couplings: tuple[float, float] = (-1, 1),
        seed: int = 42
        ):

        """ Create Dataset containing Ising data """

        np.random.seed(seed)
        xs = []
        ys = []
        theta = torch.Tensor([])
        # create coupling matrix and add noise
        gamma_data = GammaInitialization("random").initialize(
            size, value_range_couplings
        )
        self._ising_configs = []
        for i in range(num_samples):
            theta = torch.Tensor(
                np.random.uniform(
                    value_range_biases[0], value_range_biases[1], size
                )
            )
            xs.append(theta)
            sample_set = ExactSolver().sample_ising(
                utils.vector_to_biases(theta),
                utils.gamma_to_couplings(gamma_data),
            )
            sample_set.resolve()
            ys.append(sample_set.first.energy)
            self._ising_configs.append(
                np.array(list(sample_set.first.sample.values()))
            )
        self.x = torch.stack(xs)
        self.y = torch.Tensor(ys)
        self.len = len(self.y)
        self.data_size = len(theta)
        self._gamma_data = gamma_data.copy()

    def create_data_fun(
        self, function: callable, num_samples: int, ranges: list
    ):
        """ Create Dataset containing data from a given function"""

        xs = []
        ys = []
        for i in range(num_samples):
            x = [
                np.random.uniform(value_range[0], value_range[1])
                for value_range in ranges
            ]
            xs.append(torch.Tensor(x))
            try:
                ys.append(function(*x))
            except TypeError:
                msg = "number of arguments in function does not match number of ranges"
                raise TypeError(msg)
        self.x = torch.stack(xs)
        self.y = torch.Tensor(ys)
        self.len = len(self.y)
        self.data_size = len(x)

    def create_data_rand(self,
                         size: int,
                         num_samples: int,
                         value_range_biases: tuple[float, float] = (-1, 1),
                         value_range_energies: tuple[float, float] = (-1, 0),
                         seed: int = 42,
                         ):
        np.random.seed(seed)
        self.x = torch.Tensor(np.random.uniform(value_range_biases[0], value_range_biases[1], (num_samples, size)))
        self.y = torch.Tensor(np.random.uniform(value_range_energies[0], value_range_energies[1], num_samples))
        self.len = len(self.y)
        self.data_size = size


    @staticmethod
    def create_bas(stripes: bool, size: int):
        """ Create Dataset containing bars and stripes data encodes in a matrix with 0/1 entries"""
        matrix = np.zeros((size,size))
        found = False
        while not found:
            one_rows = 0
            for i in range(size):
                make_ones = bool(np.random.randint(0,2))
                if make_ones:
                    one_rows += 1
                    matrix[i,:] = np.ones(size)
            if one_rows != 0 and one_rows != size:
                found = True
        if stripes:
            return matrix.T.flatten()
        else:
            return matrix.flatten()


    def create_data_bas(self,
                        size : int,
                        num_samples : int,
                        multiplier : float = 1):
        """
        Create Dataset containing bars and stripes data encodes in a matrix with 0/1 entries
        store in x,y
        """
        xs = []
        ys = []
        for i in range(num_samples):
            b_or_s = np.random.randint(0,2)
            ys.append(torch.tensor(b_or_s * multiplier))
            xs.append(torch.tensor(SimpleDataset.create_bas(bool(b_or_s),size)))

        self.x = torch.stack(xs)
        self.y = torch.Tensor(ys)
        self.len = len(self.y)
        self.data_size = size ** 2



    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx

    def resize(
        self, size: int, hidden_nodes: HiddenNodesInitialization
    ) -> None:
        """Resize the dataset to the given size by adding hidden nodes."""
        if size < self.data_size:
            msg = "size must be greater or equal to the size of the dataset"
            raise ValueError(msg)
        elif size == self.data_size:
            return

        if hidden_nodes.mode == "random":
            hidden_nodes._random_range = (torch.min(self.x), torch.max(self.x))
        if hidden_nodes.mode == "function":
            if hidden_nodes.function is None:
                msg = "function must be given when mode is function"
                raise ValueError(msg)

        x_new = []
        for theta in self.x:
            if hidden_nodes.mode == "function":
                if hidden_nodes.fun_args is None:
                    x_new.append(
                        torch.Tensor(
                            [
                                hidden_nodes.function(theta, index_new)
                                for index_new in range(size)
                            ]
                        )
                    )
                else:
                    x_new.append(
                        torch.Tensor(
                            [
                                hidden_nodes.function(
                                    theta, index_new, hidden_nodes.fun_args
                                )
                                for index_new in range(size)
                            ]
                        )
                    )
            else:
                x_new.append(
                    torch.Tensor(
                        [
                            SimpleDataset._create_entry(
                                theta, index_new, hidden_nodes
                            )
                            for index_new in range(size)
                        ]
                    )
                )
        self.x = torch.stack(x_new)

    @staticmethod
    def _create_entry(
        theta: torch.Tensor,
        index_new: int,
        hidden_nodes: HiddenNodesInitialization,
    ) -> float:
        """Create a new value for the given index of the theta tensor."""
        multiple = index_new // len(theta)

        if hidden_nodes.mode == "zeros":
            if multiple == 0:
                return theta[index_new]
            else:
                return 0
        elif hidden_nodes.mode == "repeat":
            return theta[index_new % len(theta)]
        elif hidden_nodes.mode == "random":
            if multiple == 0:
                return theta[index_new]
            else:
                return np.random.uniform(
                    hidden_nodes.random_range[0], hidden_nodes.random_range[1]
                )

    @staticmethod
    def lin_scaling(
        theta: torch.Tensor, index_new: int, fun_args: tuple
    ) -> float:
        """Linear scaling of the theta vector."""
        mod = index_new % len(theta)
        mult = index_new // len(theta) + 1
        return (theta[mod] * (mult * fun_args[0]))**3 + fun_args[1]

    @staticmethod
    def offset(theta: torch.Tensor, index_new: int, fun_args: tuple) -> float:
        """
        Calculates a new value for the given index of
        the theta tensor by adding an offset.
        """
        offset = fun_args[0]
        return theta[index_new % len(theta)] + index_new // len(theta) * offset

    @staticmethod
    def offset_fixed(
        theta: torch.Tensor, index_new: int, fun_args: tuple
    ) -> float:
        """
        Calculates a new value for the given index of the theta tensor by adding
        an offset.
        """
        offset = fun_args[0]
        if index_new == 19:
            return 10000
        return theta[index_new % len(theta)] + index_new // len(theta) * offset

    @staticmethod
    def offset_random(
        theta: torch.Tensor, index_new: int, fun_args: tuple
    ) -> float:
        """
        Calculates a new value for the given index of the theta tensor by adding
        an offset.
        """
        offset = fun_args[0]
        if index_new == 0:
            return 10
        return np.random.uniform(-offset, offset)
