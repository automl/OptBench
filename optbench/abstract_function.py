from __future__ import annotations

from abc import abstractmethod

import time
from typing import TYPE_CHECKING, Any

from ConfigSpace import ConfigurationSpace, Float

import numpy as np
from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue


if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger

class AbstractFunction(ObjectiveFunction):
    def __init__(self, dim: int, lower_bounds: list[float], upper_bounds: list[float], seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(loggers=loggers)

        self.dim = dim
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Uniform Float Hyperparameters
        hps = [
            Float(name=f"x_{i}", bounds=(self.lower_bounds[i], self.upper_bounds[i]))
            for i in range(dim)
        ]
        self._configspace = ConfigurationSpace(
            space=hps,
            seed=seed
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
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