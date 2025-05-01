from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ConfigSpace import ConfigurationSpace, Float

from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialInfo, TrialValue

from optbench.abstract_function import AbstractFunction

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger



class OptBenchObjectiveFunction(ObjectiveFunction):
    def __init__(self, function: AbstractFunction, loggers: list[AbstractLogger] | None = None) -> None:
        super().__init__(loggers=loggers)

        self.function = function

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.function.configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return self.function.evaluate(trial_info)

    def f_min(self) -> float | None:
        return self.function.f_min()