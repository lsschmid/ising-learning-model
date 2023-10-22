"""QPU model for ising_learning_model."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np
from ising_learning_model.model import Model
from ising_learning_model.utils import utils
from dimod import SampleSet
from dwave.system import (
    DWaveSampler,
    EmbeddingComposite,
    FixedEmbeddingComposite,
)
from networkx import complete_graph
from multiprocessing import Pool
from time import perf_counter


def _sample_bulk_threaded(
    parameters: list[
        torch.Tensor | int | dict | str | bool | FixedEmbeddingComposite
    ],
) -> tuple[list, list, list]:
    """
    Sample the model in bulk using extra thread to free required memory afterwards.
    Otherwise, the large number of samples may cause memory issues.
    """
    (
        thetas,
        ys,
        gamma,
        lmd,
        offset,
        num_reads,
        embedding,
        sampler,
        _verbose,
        _save_params,
        loss,
    ) = parameters
    if isinstance(sampler, str):
        sampler = FixedEmbeddingComposite(
            DWaveSampler(profile=sampler), embedding
        )

    losses_bulk = []
    energies_bulk = []
    configurations_bulk = []
    for idx, (theta, y) in enumerate(zip(thetas, ys)):
        if _verbose:
            print(f"\t\tsample {idx + 1}/{len(thetas)}")
        # create the Ising model
        h = utils.vector_to_biases(theta)
        J = utils.gamma_to_couplings(gamma)
        t_before = perf_counter()
        sample_set = sampler.sample_ising(h, J, num_reads=num_reads)
        # wait for the sample_set to be resolved
        sample_set.resolve()
        if _verbose:
            t_after = perf_counter()
            print(f"\t\tsampled in {t_after - t_before} seconds")
        energy = lmd * sample_set.first.energy + offset
        # energy = sample_set.first.energy
        losses_bulk.append(loss(energy, y))
        if _save_params:
            energies_bulk.append(energy)
            configurations_bulk.append(
                np.array(list(sample_set.first.sample.values()))
            )
    return losses_bulk, energies_bulk, configurations_bulk


class QPUModel(Model):
    _num_reads: int
    _embedding: dict
    _sampler: FixedEmbeddingComposite
    _profile: str

    def __init__(
        self, size: int, profile: str = "default", num_reads: int = 1
    ):
        super().__init__(size)
        # Find embedding once and use it for all samples
        print("Searching QPU and computing embedding...")
        toy_sampler = EmbeddingComposite(DWaveSampler(profile=profile))
        self._embedding = toy_sampler.find_embedding(
            complete_graph(size).edges(), toy_sampler.child.edgelist
        )
        print("Embedding found.")
        self._sampler = FixedEmbeddingComposite(
            DWaveSampler(profile=profile), self._embedding
        )
        print(f"using QPU: {self._sampler.child.solver.id}")
        self._num_reads = num_reads
        self._profile = profile

    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(h, j, num_reads=self._num_reads)

    def _sample_bulk(
        self, thetas: torch.Tensor, ys: torch.Tensor
    ) -> tuple[list, list, list]:
        """
        Sample the ising model in bulk.
        """
        parameters = [
            thetas,
            ys,
            self._gamma,
            self._lmd,
            self._offset,
            self._num_reads,
            self._embedding,
            self._profile,
            self._verbose,
            self._save_params,
            self.loss_function,
        ]
        if self.size > 30 and len(thetas) > 1:
            with Pool(1) as pool:
                if self._verbose:
                    print("\t\tsampling in bulk using extra thread...")
                return pool.apply(_sample_bulk_threaded, [parameters])
        else:
            return _sample_bulk_threaded(parameters)
