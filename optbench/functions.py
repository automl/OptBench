from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from numpy import ndarray
from abc import abstractmethod

from carps.loggers.abstract_logger import AbstractLogger

from optbench.abstract_function import AbstractFunction


class Ackley(AbstractFunction):
    def __init__(
        self,
        dim: int,
        seed: int | None = None,
        a: float = 20,
        b: float = 0.2,
        c: float = 2 * np.pi,
        loggers: list[AbstractLogger] | None = None,
    ) -> None:
        lower_bounds = np.array([-32.768] * dim)
        upper_bounds = -lower_bounds

        super().__init__(dim, lower_bounds, upper_bounds, seed=seed, loggers=loggers)

        self.a = a
        self.b = b
        self.c = c

        self._configspace = ConfigurationSpace(
            # Do not use 0 as default because this is the global minimum
            {
                f"x_{i}": Float(bounds=(self.lower_bounds[i], self.upper_bounds[i]), default=-10, name=f"x_{i}")
                for i in range(dim)
            },
            seed=seed,
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
        return (
            -self.a * np.exp(-self.b * np.sqrt(np.inner(x, x) / len(x)))
            - np.exp(np.cos(self.c * x).sum() / len(x))
            + self.a
            + np.e
        )


class Levy(AbstractFunction):
    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        lower_bounds = np.array([-10] * dim)
        upper_bounds = -lower_bounds

        super().__init__(dim, lower_bounds, upper_bounds, seed=seed, loggers=loggers)

        self._configspace = ConfigurationSpace(
            {
                f"x_{i}": Float(bounds=(self.lower_bounds[i], self.upper_bounds[i]), default=-4, name=f"x_{i}")
                for i in range(dim)
            },
            seed=seed,
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
        term3 = ((w[d - 1] - 1) ** 2) * (1 + np.sin(2 * np.pi * w[d - 1]) ** 2)
        wi = w[0 : d - 2]
        Sum = np.sum(((wi - 1) ** 2) * (1 + 10 * np.sin(np.pi * wi + 1) ** 2))
        return term1 + Sum + term3


class Schwefel(AbstractFunction):
    def __init__(self, dim: int, seed: int | None = None, loggers: list[AbstractLogger] | None = None) -> None:
        lower_bounds = np.array([-500] * dim)
        upper_bounds = -lower_bounds

        super().__init__(dim, lower_bounds, upper_bounds, seed=seed, loggers=loggers)

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
        return 418.9829 * d - Sum


class Hartmann(AbstractFunction):
    """Hartmann.

    Scaled: Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark of kriging-based infill criteria for noisy optimization.
    Function has zero mean and variance of one.
    """

    def __init__(
        self, dim: int, scaled: bool = False, seed: int | None = None, loggers: list[AbstractLogger] | None = None
    ) -> None:
        assert dim in [3, 4, 6], f"Hartmann is only defined for 3, 4 and 6 dimensions, but got {dim}."
        lower_bounds = np.array([0] * dim)
        upper_bounds = np.array([1] * dim)

        self._scaled = scaled

        self._set_matrices()

        super().__init__(dim, lower_bounds, upper_bounds, seed=seed, loggers=loggers)

    @abstractmethod
    def _set_matrices(self):
        self._A = ...
        self._P = ...
        self._alpha = ...

    def _function(self, x: ndarray) -> float:
        # https://www.sfu.ca/~ssurjano/Code/hart6.html
        alpha = self._alpha
        A = self._A
        P = self._P

        xx = np.asarray(x).reshape(-1)

        # Repeat input vector to match dimensions
        xxmat = np.tile(xx, (4, 1))  # 4x6

        # Compute inner term
        inner = np.sum(A * (xxmat - P) ** 2, axis=1)

        # Compute outer sum
        outer = np.sum(alpha * np.exp(-inner))

        return -outer


class Hartmann3(Hartmann):
    """Hartmann3."""

    def __init__(
        self, dim: int = 3, scaled: bool = False, seed: int | None = None, loggers: list[AbstractLogger] | None = None
    ) -> None:
        super().__init__(dim, scaled, seed, loggers)

    @property
    def x_min(self) -> ndarray | None:
        return np.array([0.114614, 0.555649, 0.852547])

    @property
    def f_min(self) -> float | None:
        return -3.86278

    def _set_matrices(self):
        self._alpha = np.array([1.0, 1.2, 3, 3.2])

        self._A = np.array([[3, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        self._P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])


class Hartmann6(Hartmann):
    """Hartmann6.

    Scaled: Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark of kriging-based infill criteria for noisy optimization.
    Function has zero mean and variance of one.
    """

    def __init__(
        self, dim: int = 6, scaled: bool = False, seed: int | None = None, loggers: list[AbstractLogger] | None = None
    ) -> None:
        super().__init__(dim, scaled, seed, loggers)

    @property
    def x_min(self) -> ndarray | None:
        if not self._scaled:
            return np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        return None

    @property
    def f_min(self) -> float | None:
        if not self._scaled:
            return -3.32237
        return None

    def _set_matrices(self):
        self._alpha = np.array([1.0, 1.2, 3, 3.2])

        self._A = np.array(
            [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
        )
        self._P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

    def _function(self, x: ndarray) -> float:
        outer = super()._function(x)

        if self._scaled:
            return -(2.58 - outer) / 1.94
        return outer


class Hartmann4(Hartmann6):
    """Hartmann6.

    Scaled: Picheny, V., Wagner, T., & Ginsbourger, D. (2012). A benchmark of kriging-based infill criteria for noisy optimization.
    Function has zero mean and variance of one.
    """

    def __init__(
        self, dim: int = 4, scaled: bool = False, seed: int | None = None, loggers: list[AbstractLogger] | None = None
    ) -> None:
        super().__init__(dim, scaled, seed, loggers)

    @property
    def x_min(self) -> ndarray | None:
        return None

    @property
    def f_min(self) -> float | None:
        return None

    def _function(self, x: ndarray) -> float:
        outer = super()._function(x)
        return -(1.1 - outer) / 0.839
