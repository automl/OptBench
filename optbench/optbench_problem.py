from __future__ import annotations

from typing import TYPE_CHECKING

from smacbenchmarking.benchmarks.problem import Problem

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from smacbenchmarking.utils.trials import TrialInfo, TrialValue

    from optbench.abstract_function import AbstractFunction
    from smacbenchmarking.loggers.abstract_logger import AbstractLogger



class OptBenchProblem(Problem):
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