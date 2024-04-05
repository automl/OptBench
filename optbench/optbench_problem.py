from __future__ import annotations

from typing import TYPE_CHECKING

from smacbenchmarking.benchmarks.problem import Problem

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from smacbenchmarking.utils.trials import TrialInfo, TrialValue

    from optbench.abstract_function import AbstractFunction


class OptBenchProblem(Problem):
    def __init__(self, function: AbstractFunction) -> None:
        super().__init__()

        self.function = function

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.function.configspace

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return self.function.evaluate(trial_info)

    def f_min(self) -> float | None:
        return self.function.f_min()