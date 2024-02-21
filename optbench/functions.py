from __future__ import annotations

import numpy as np
from numpy import ndarray

from optbench.abstract_function import AbstractFunction

from ConfigSpace import ConfigurationSpace, Float

class Ackley(AbstractFunction):
    def __init__(self, dim: int, a: float = 20, b: float = 0.2, c: float = 2 * np.pi) -> None:
        lower_bounds = np.array([-32.768] * dim)
        upper_bounds = -lower_bounds 

        super().__init__(dim, lower_bounds, upper_bounds)

        self.a = a
        self.b = b
        self.c = c

        self._configspace = ConfigurationSpace(
            # Do not use 0 as default because this is the global minimum
            {f"x_{i}": Float(bounds=(self.lower_bounds[i], self.upper_bounds[i]), default=-10, name=f"x_{i}") for i in range(dim)}
        )

    @property
    def x_min(self) -> ndarray | None:
        return np.zeros((self.dim,))
    
    @property
    def f_min(self) -> float | None:
        return 0

    def _function(self, x: ndarray) -> float:
        # https://www.sfu.ca/~ssurjano/Code/ackleyr.html
        x = np.asarray(x).reshape(-1)
        result = (-self.a * np.exp(-self.b * np.sqrt(np.inner(x, x) / len(x))) - np.exp(
            np.cos(self.c * x).sum() / len(x)) + self.a + np.e)
        return result
       

class Levy(AbstractFunction):
    def __init__(self, dim: int) -> None:
        lower_bounds = np.array([-10] * dim)
        upper_bounds = -lower_bounds 

        super().__init__(dim, lower_bounds, upper_bounds)

        self._configspace = ConfigurationSpace(
            {f"x_{i}": Float(bounds=(self.lower_bounds[i], self.upper_bounds[i]), default=-4, name=f"x_{i}") for i in range(dim)}
        )

    @property
    def x_min(self) -> ndarray | None:
        return np.ones((self.dim,))
    
    @property
    def f_min(self) -> float | None:
        return 0

    def _function(self, x: ndarray) -> float:
        # https://www.sfu.ca/~ssurjano/Code/levyr.html
        x = np.asarray(x).reshape(-1)
        d = len(x)
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term3 = ((w[d-1] - 1) ** 2) * (1 + np.sin(2*np.pi * w[d-1]) ** 2)
        wi = w[0:d-2]
        Sum = np.sum(((wi - 1)**2) * (1 + 10*np.sin(np.pi * wi + 1) ** 2 ))
        y = term1 + Sum + term3
        return y
    
class Schwefel(AbstractFunction):
    def __init__(self, dim: int) -> None:
        lower_bounds = np.array([-500] * dim)
        upper_bounds = -lower_bounds 

        super().__init__(dim, lower_bounds, upper_bounds)

    @property
    def x_min(self) -> ndarray | None:
        return np.ones((self.dim,)) * 420.9687
    
    @property
    def f_min(self) -> float | None:
        return 0

    def _function(self, x: ndarray) -> float:
        # https://www.sfu.ca/~ssurjano/Code/schwefr.html
        x = np.asarray(x).reshape(-1)
        d = len(x)
        Sum = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        y = 418.9829 * d - Sum
        return y
       