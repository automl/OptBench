#!/usr/bin/env python

"""Tests for `optbench` package."""

import inspect

import pytest
import numpy as np

from optbench import functions
from optbench.abstract_function import AbstractFunction
from optbench.functions import Schwefel

from omegaconf import OmegaConf
from hydra.utils import instantiate
from carps.utils.trials import TrialInfo

function_classes = [c[1] for c in inspect.getmembers(functions, inspect.isclass) if issubclass(c[1], AbstractFunction) and c[1] != AbstractFunction]

@pytest.mark.parametrize("funcclass", function_classes)
def test_opt(funcclass):
    func = funcclass(dim=4)
    y = func._function(func.x_min)
    if funcclass == Schwefel:
        # Schwefel optimum is weird although function is implemented according to source
        assert np.isclose(func.f_min, y, atol=1e-4)
    else:
        assert np.isclose(func.f_min, y)

@pytest.mark.parametrize("path", ["optbench/configs/problem/OptBench/Ackley_5.yaml", "optbench/configs/problem/OptBench/Levy_5.yaml", "optbench/configs/problem/OptBench/Schwefel_5.yaml"])
def test_instantiate_and_evaluate(path: str):
    cfg = OmegaConf.load(path)
    cfg.problem.function.seed = 1
    problem = instantiate(cfg.problem)
    problem.evaluate(TrialInfo(config=problem.configspace.sample_configuration()))
