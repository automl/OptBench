from __future__ import annotations

from smacbenchmarking.benchmarks.problem import Problem
from ConfigSpace import ConfigurationSpace
from optbench.abstract_function import AbstractFunction
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

class OptBenchProblem(Problem):
    def __init__(self, function: AbstractFunction) -> None:
        super().__init__()

        self.function = function

    def configspace(self) -> ConfigurationSpace:
        return self.function.configspace
    
    def evaluate(self, trial_info: TrialInfo) -> TrialValue:
        return self.function.evaluate(trial_info)
    
    def f_min(self) -> float | None:
        return self.function.f_min()