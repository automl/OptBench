#!/usr/bin/env python

"""Tests for `optbench` package."""

import pytest


from optbench import functions

from optbench import functions
from optbench.abstract_function import AbstractFunction
import inspect

classes = [c[1] for c in inspect.getmembers(functions, inspect.isclass) if issubclass(c[1], AbstractFunction)]

@pytest.mark.parametrize("funcclass", classes)
def test_opt(funcclass):
    func = funcclass(dim=4)
    y = func._function(func.x_min)
    assert func.f_min == y



