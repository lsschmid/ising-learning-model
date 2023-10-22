""" Simulated Annealing Model"""
from __future__ import annotations
from ising_learning_model.model import Model
from dimod import SampleSet
from neal import SimulatedAnnealingSampler


class AnnealingSettings:
    """Settings for the simulated annealing sampler."""
    beta_range: list
    num_reads: int
    num_sweeps: int
    num_sweeps_per_beta: int
    beta_schedule_type: str

    def __init__(self) -> None:
        self.beta_range = None
        self.num_reads = 1
        self.num_sweeps = 100
        self.num_sweeps_per_beta = 1
        self.beta_schedule_type = "geometric"


class SimAnnealModel(Model):
    """Solves the provided Ising problem using the simulated annealing sampler from DWAVE neal."""
    def __init__(
        self, size: int, settings: AnnealingSettings = AnnealingSettings()
    ):
        super().__init__(size)
        self._sampler = SimulatedAnnealingSampler()
        self._embedding = None
        self.annealing_settings = settings

    def _sample_single(self, h: dict, j: dict) -> SampleSet:
        return self._sampler.sample_ising(
            h,
            j,
            beta_range=self.annealing_settings.beta_range,
            num_reads=self.annealing_settings.num_reads,
            num_sweeps=self.annealing_settings.num_sweeps,
            num_sweeps_per_beta=self.annealing_settings.num_sweeps_per_beta,
            beta_schedule_type=self.annealing_settings.beta_schedule_type,
        )
