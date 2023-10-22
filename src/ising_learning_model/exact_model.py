"""Exact model."""
from __future__ import annotations
from ising_learning_model.model import Model
from dimod import ExactSolver, SampleSet


class ExactModel(Model):
    """Solves the provided Ising problem exactly using the dimod ExactSolver."""
    def __init__(self, size: int):
        super().__init__(size)
        self._sampler = ExactSolver()
        self._embedding = None

    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(h, j)
