#!/usr/bin/env python

"""Tests for `optbench` package."""

import inspect

import pytest

from optbench import functions
from optbench.abstract_function import AbstractFunction

function_classes = [c[1] for c in inspect.getmembers(functions, inspect.isclass) if issubclass(c[1], AbstractFunction) and c[1] != AbstractFunction]

@pytest.mark.parametrize("funcclass", function_classes)
def test_opt(funcclass):
    func = funcclass(dim=4)
    y = func._function(func.x_min)
    assert func.f_min == y



from omegaconf import OmegaConf
from hydra.utils import instantiate
from smacbenchmarking.utils.trials import TrialInfo, TrialValue

path = "../lib/OptBench/optbench/configs/problem/OptBench/Ackley_5.yaml"
cfg = OmegaConf.load(path)
cfg.problem.function.seed = 1
problem = instantiate(cfg.problem)
problem.evaluate(TrialInfo(config=problem.configspace().sample_configuration()))
