"""Abstract class for the model."""
from __future__ import annotations
from abc import ABC, abstractmethod

import pandas as pd
import torch
import numpy as np
import pickle

from dimod import SampleSet

from time import perf_counter
from torch.utils.data import DataLoader
from ising_learning_model.utils import GammaInitialization, utils
from ising_learning_model.data import SimpleDataset, HiddenNodesInitialization


class ModelSetting:
    """Model settings for the model."""

    gamma_init: GammaInitialization
    hidden_nodes_init: HiddenNodesInitialization
    mini_batch_size: int
    num_reads: int
    optim_steps: int
    learning_rate: float
    learning_rate_lmd: float
    dacay_rate: float

    def __init__(self) -> None:
        self.gamma_init = GammaInitialization("zeros")
        self.hidden_nodes_init = HiddenNodesInitialization("repeat")
        self.mini_batch_size = 1
        self.num_reads = 1
        self.optim_steps = 1
        self.learning_rate = 0.1
        self.learning_rate_lmd = 0.1
        self.dacay_rate = 1


class ModelResults:
    """ Class to store the results of the model"""
    results: pd.DataFrame
    results_samples: pd.DataFrame
    runtime: float
    final_loss: float

    def __init__(self, results, results_samples, runtime, final_loss, results_test=None, results_samples_test=None):
        self.results = results
        self.results_samples = results_samples
        self.runtime = runtime
        self.final_loss = final_loss
        self.results_test = results_test
        self.results_samples_test = results_samples_test


class Model(ABC):
    """
    Abstract class for the model. The derived classes use differnt Ising machines to sample the Ising model.
    In particular, the derived classes implement the _sample_single method, which samples the Ising model
    using the respective Ising machine.
    Examples for derived classes are ExactModel, SimulatedAnnealingModel, DwaveSamplerModel, which use
    the dimod ExactSolver, dimod SimulatedAnnealingSampler and the D-Wave sampler, respectively.
    """

    size: int
    lmd_init_value: float
    offset_init_value: float
    settings: ModelSetting
    loss_function: callable
    _gamma: np.array
    _lmd: float
    _offset: float
    _verbose: bool
    _save_params: bool
    _save_samples: bool
    _sampler: any
    _embedding: any

    def __init__(self, size: int):
        """
        Initialize the model.

        param size: the size of the Ising model
        type size: int
        """
        self._training_set = None
        self._save_params = False
        self._save_samples = False
        self.size = size
        self.lmd_init_value = 1
        self.offset_init_value = 0
        self._offset = 0
        self.loss_function = utils.mse_loss_function
        self.settings = ModelSetting()
        self._gamma = np.zeros((size, size))
        self._lmd = self.lmd_init_value

    def eval_single(self, theta: np.array) -> SampleSet:
        """
        Evaluate the model.

        param theta: the biases
        type theta: np.array

        return: the sample set
        rtype: SampleSet
        """
        h = utils.vector_to_biases(theta)
        J = utils.gamma_to_couplings(self._gamma)
        return self._sample_single(h, J)

    @abstractmethod
    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        """
        Sample the Ising model.

        param h: the biases
        type h: dict
        param j: the couplings
        type j: dict

        return: the sample set
        rtype: SampleSet
        """
        ...

    def _sample_bulk(
            self, thetas: torch.Tensor, ys: torch.Tensor
    ) -> tuple[list, list, list]:
        """Sample the Ising model in bulk."""
        losses_bulk = []
        energies_bulk = []
        configurations_bulk = []
        for idx, (theta, y) in enumerate(zip(thetas, ys)):
            if self._verbose:
                print(f"\t\tsample {idx + 1}/{len(thetas)}")
            # create the Ising model
            h = utils.vector_to_biases(theta)
            J = utils.gamma_to_couplings(self._gamma)
            t_before = perf_counter()
            sample_set = self._sample_single(h, J)
            # wait for the sample_set to be resolved
            sample_set.resolve()
            if self._verbose:
                t_after = perf_counter()
                print(f"\t\tsampled in {t_after - t_before} seconds")
            energy = self._lmd * sample_set.first.energy + self._offset
            losses_bulk.append(self.loss_function(energy, y))
            energies_bulk.append(energy)
            configurations_bulk.append(
                np.array(list(sample_set.first.sample.values()))
            )
        return losses_bulk, energies_bulk, configurations_bulk

    def train(
            self,
            training_set: SimpleDataset,
            test_set: SimpleDataset = None,
            verbose: bool = False,
            save_params: bool = False,
            save_samples: bool = False,
    ) -> ModelResults:
        """
        Train the model.

        @param training_set: the dataset to train on
        @type training_set: IsingDataset
        @param verbose: whether to print progress
        @type verbose: bool
        @param save_params: whether to save the parameters like gamma, lambda and offset
        @type save_params: bool
        @param save_samples: whether to save information about the samples
        @type save_samples: bool

        @return: the model results
        @rtype: ModelResults
        """
        if training_set[0][0].shape[0] > self.size:
            msg = "dataset size does not match model size"
            raise ValueError(msg)
        self._training_set = training_set
        self._test_set = test_set

        # initialize parameters and hyperparameters
        self._gamma = self.settings.gamma_init.initialize(self.size)
        self._lmd = self.lmd_init_value
        eta = self.settings.learning_rate
        eta_lmd = self.settings.learning_rate_lmd
        self._verbose = verbose
        self._save_params = save_params
        self._save_samples = save_samples

        # resize input to match the model size
        training_set.resize(self.size, self.settings.hidden_nodes_init)

        if self.offset_init_value == "sample":
            self._offset = self._sample_offset(training_set)
            if self._verbose:
                print(f"offset: {self._offset}")
        else:
            self._offset = self.offset_init_value

        train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.settings.mini_batch_size,
            shuffle=True,
        )
        train_loader_test = None
        if test_set is not None:
            train_loader_test = torch.utils.data.DataLoader(
                test_set,
                batch_size=len(test_set.y),
                shuffle=False,
            )

        results = pd.DataFrame()
        results_samples = pd.DataFrame()
        results_test = pd.DataFrame()
        results_samples_test = pd.DataFrame()

        # start timer
        t_before = perf_counter()

        try:
            for epoch in range(self.settings.optim_steps):
                eta *= self.settings.dacay_rate
                if self._verbose:
                    print(f"epoch {epoch}/{self.settings.optim_steps}")
                # iterate over the mini batches
                for idx_mini_batch, (thetas, ys, indices) in enumerate(
                        train_loader
                ):
                    if self._verbose:
                        print(f"\tmini batch {idx_mini_batch}/{len(train_loader)}")
                    # run model on the batch
                    (
                        losses_bulk,
                        energies_bulk,
                        configurations_bulk,
                    ) = self._sample_bulk(thetas, ys)

                    # save results in a dataframe
                    if save_samples:
                        for idx_bulk, idx_theta in enumerate(indices):
                            result_sample_row = {
                                "epoch": epoch,
                                "idx_theta": int(idx_theta),
                                "idx_mini_batch": idx_mini_batch,
                                "energy": energies_bulk[idx_bulk],
                                "config": configurations_bulk[idx_bulk],
                                "target": ys[idx_bulk].item(),
                            }
                            results_samples = pd.concat(
                                [
                                    results_samples,
                                    pd.DataFrame([result_sample_row]),
                                ],
                                ignore_index=True,
                            )

                    # update the parameters
                    delta = np.mean(
                        [
                            self._lmd * (f - y) * np.outer(c, c)
                            for f, y, c in zip(
                            energies_bulk, ys.tolist(), configurations_bulk
                        )
                        ],
                        axis=0,
                    )
                    # delta_lmd = np.mean([(f - y) * ((f - self._offset) /
                    # self._lmd) for f, y in zip(energies_bulk, ys.tolist())])
                    delta_lmd = (
                                        training_set.y.max().numpy() - training_set.y.min().numpy()
                                ) / (np.max(energies_bulk) - np.min(energies_bulk))

                    self._gamma -= utils.make_upper_triangular(eta * delta)
                    self._lmd -= eta_lmd * delta_lmd

                    loss = np.mean(losses_bulk)
                    energy = np.mean(energies_bulk)
                    print(loss)
                    if verbose:
                        print(loss)
                    result_row = {
                        "epoch": epoch,
                        "idx_minibatch": idx_mini_batch,
                        "loss": loss,
                        "offset": self._offset,
                    }

                    # only on demand as gamma can be large
                    if save_params:
                        result_row["gamma"] = self._gamma.copy()
                        result_row["lmd"] = self._lmd
                    results = pd.concat(
                        [results, pd.DataFrame([result_row])], ignore_index=True
                    )

                # test the model -----------------------------
                if test_set is not None:
                    print("sampling test set")
                    for idx_mini_batch, (thetas, ys, indices) in enumerate(
                            train_loader_test
                    ):
                        # run model on the batch
                        (
                            losses_bulk,
                            energies_bulk,
                            configurations_bulk,
                        ) = self._sample_bulk(thetas, ys)

                        # save results in a dataframe
                        if save_samples:
                            for idx_bulk, idx_theta in enumerate(indices):
                                result_sample_row = {
                                    "epoch": epoch,
                                    "idx_theta": int(idx_theta),
                                    "idx_mini_batch": idx_mini_batch,
                                    "energy": energies_bulk[idx_bulk],
                                    "config": configurations_bulk[idx_bulk],
                                    "target": ys[idx_bulk].item(),
                                }
                                results_samples_test = pd.concat(
                                    [
                                        results_samples_test,
                                        pd.DataFrame([result_sample_row]),
                                    ],
                                    ignore_index=True,
                                )

                        loss = np.mean(losses_bulk)
                        energy = np.mean(energies_bulk)
                        result_row_test = {
                            "epoch": epoch,
                            "idx_minibatch": idx_mini_batch,
                            "loss": loss,
                            "offset": self._offset,
                        }
                        results_test = pd.concat(
                            [results_test, pd.DataFrame([result_row_test])], ignore_index=True
                        )
        # catch any exception and exit training
        except Exception as e:
            print(e)
            t_after = perf_counter()
            t_total = (t_after - t_before) / 60
            return ModelResults(
                results, results_samples, t_total, results["loss"].iloc[-1],
                results_test, results_samples_test
            )

        t_after = perf_counter()
        t_total = (t_after - t_before) / 60

        return ModelResults(
            results, results_samples, t_total, results["loss"].iloc[-1],
            results_test, results_samples_test
        )

    def test(
            self, test_set: SimpleDataset, gammas: list[np.array] = None, save_samples: bool = False
    ) -> ModelResults:
        """
        Evaluate the model on the test set.
        :param test_set: dataset to evaluate on
        :type test_set: IsingDataset
        :param gammas: list of gammas to evaluate.
        If None, the gamma from the trained Model is used
        :type gammas: list[np.array]

        :return: ModelResults
        """
        if gammas is None:
            gammas = [self._gamma]
        results = pd.DataFrame()
        results_samples = pd.DataFrame()
        test_loader = DataLoader(test_set)

        t_before = perf_counter()
        for idx_gamma, gamma in enumerate(gammas):
            if self._verbose:
                print(f"gamma {idx_gamma}/{len(gammas)}")

            for idx, (thetas, ys, indices) in enumerate(test_loader):
                (
                    losses_bulk,
                    energies_bulk,
                    configurations_bulk,
                ) = self._sample_bulk(thetas, ys)

                for idx_bulk, idx_theta in enumerate(indices):
                    config = configurations_bulk[idx_bulk] if save_samples else None
                    result_sample_row = {
                        "epoch": idx_gamma,
                        "idx_theta": int(idx_theta),
                        "energy": energies_bulk[idx_bulk],
                        "config": config,
                        "target": ys[idx_bulk].item(),
                    }
                    results_samples = pd.concat(
                        [results_samples, pd.DataFrame([result_sample_row])],
                        ignore_index=True,
                    )

                loss = np.mean(losses_bulk)
                result_row = {"gamma_idx": idx_gamma, "loss_test": loss}
                results = pd.concat(
                    [results, pd.DataFrame([result_row])], ignore_index=True
                )

        t_after = perf_counter()
        t_total = (t_after - t_before) / 60
        return ModelResults(
            results, results_samples, t_total, results["loss_test"].iloc[-1]
        )

    def _sample_offset(self, training_set: SimpleDataset) -> float:
        """
        Sample the offset from the training set.
        :param training_set: training set
        :type training_set: IsingDataset

        :return: offset
        :rtype: float
        """
        energies = []
        for theta in training_set.x:
            sample_set = self.eval_single(theta)
            energies.append(self._lmd * sample_set.first.energy)
        return -np.mean(energies) + training_set.y.mean().numpy()

    def _save_model(self, path: str):
        """
        Save the model to a file.
        :param path: path to save the model to
        :type path: str
        """
        params = {
            "gamma": self._gamma,
            "lmd": self._lmd,
            "offset": self._offset,
            "training_set": self._training_set,
        }
        with open(path, "wb") as f:
            pickle.dump(params, f)

    def _load_model(self, path: str):
        """
        Load the model from a file.
        :param path: path to load the model from
        :type path: str
        """
        with open(path, "rb") as f:
            params = pickle.load(f)
        self._gamma = params["gamma"]
        self._lmd = params["lmd"]
        self._offset = params["offset"]
        self._training_set = params["training_set"]
