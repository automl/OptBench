from __future__ import annotations

from abc import abstractmethod

import numpy as np
from ConfigSpace import ConfigurationSpace
from smacbenchmarking.benchmarks.problem import Problem
from smacbenchmarking.utils.trials import TrialInfo, TrialValue


class AbstractFunction(Problem):
    def __init__(self, dim: int, lower_bounds: list[float], upper_bounds: list[float], seed: int | None = None) -> None:
        super().__init__()

        self.dim = dim
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Uniform Float Hyperparameters
        space = {f"x_{i}": (self.lower_bounds[i], self.upper_bounds[i]) for i in range(dim)}
        self._configspace = ConfigurationSpace(
            space=space,
            seed=seed
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        config = trial_info.config
        x = np.array(list(config.values()))
        cost = self._function(x=x)
        return TrialValue(cost=cost)

    @abstractmethod
    def _function(x: np.ndarray) -> float:
        ...

    @property
    def x_min(self) -> np.ndarray | None:
        """Return the configuration with the minimum function value.

        Returns:
        -------
        np.ndarray | None
            Point with minimum function value (if exists).
            Else, return None.
        """
        return None