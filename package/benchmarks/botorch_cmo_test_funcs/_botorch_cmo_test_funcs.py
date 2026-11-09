"""
This code is taken from botorch (https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py)
distributed under the MIT License.

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

OptunaHub ``BaseProblem`` wrappers for 6 of the problems from ``botorch``.

``ConstrainedBraninCurrin`` and ``MW7`` read their inputs as ``numpy.float64``
rather than plain ``float``: both have a division that is well-defined at a
bound (Currin's ``1 / x1`` at ``x1 = 0``; MW7's ``f1 / f0`` at ``x0 = 0``) only
under IEEE 754 semantics (``+inf`` rather than a ``ZeroDivisionError``), which
is what BoTorch's ``numpy``-backed originals -- and the golden values recorded
from them -- rely on.

Problems: BNH, CONSTR, ConstrainedBraninCurrin, OSY, SRN, MW7.

Conventions
-----------
- Parameters are named ``x0``, ``x1``, ... following each problem's input
  vector order.
- ``evaluate_constraints`` keys are named ``c0``, ``c1``, ... following each
  problem's constraint order.
- BoTorch's constraint convention is a slack ``c_i(x) >= 0`` is feasible;
  OptunaHub's ``BaseProblem`` convention is a value ``<= 0`` is feasible. So
  every constraint below is the *negation* of the corresponding BoTorch slack.
- ``OSY``: BoTorch's raw ``_evaluate_true`` already computes the original
  Osyczka & Kundu (1995) minimization objectives verbatim (``f0`` is
  defined with its minus sign baked into the formula); BoTorch's own
  ``negate=True`` recommendation is about fitting its own maximize-oriented
  ``ref_point``/hypervolume convention, not about the sign of ``f0`` itself.
  So this wrapper uses the raw values unchanged with
  ``directions=[MINIMIZE, MINIMIZE]``.
"""  # NOQA

from __future__ import annotations

import math
from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
from optunahub.benchmarks import BaseProblem


_MINIMIZE_2D = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]


def _box_search_space(bounds: list[tuple[float, float]]) -> dict[str, BaseDistribution]:
    return {f"x{i}": FloatDistribution(low=lo, high=hi) for i, (lo, hi) in enumerate(bounds)}


class BNH(BaseProblem):
    """The constrained BNH problem. See [GarridoMerchan2020]_. Minimization."""

    _bounds = [(0.0, 5.0), (0.0, 3.0)]

    @property
    def search_space(self) -> dict[str, BaseDistribution]:
        return _box_search_space(self._bounds)

    @property
    def directions(self) -> list[StudyDirection]:
        return _MINIMIZE_2D

    def evaluate(self, params: dict[str, Any]) -> list[float]:
        x0, x1 = params["x0"], params["x1"]
        f0 = 4.0 * (x0**2 + x1**2)
        f1 = (x0 - 5.0) ** 2 + (x1 - 5.0) ** 2
        return [f0, f1]

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        x0, x1 = params["x0"], params["x1"]
        c0 = (x0 - 5.0) ** 2 + x1**2 - 25.0
        c1 = 7.7 - (x0 - 8.0) ** 2 - (x1 + 3.0) ** 2
        return {"c0": c0, "c1": c1}


class CONSTR(BaseProblem):
    """The constrained CONSTR problem. See [GarridoMerchan2020]_. Minimization."""

    _bounds = [(0.1, 10.0), (0.0, 5.0)]

    @property
    def search_space(self) -> dict[str, BaseDistribution]:
        return _box_search_space(self._bounds)

    @property
    def directions(self) -> list[StudyDirection]:
        return _MINIMIZE_2D

    def evaluate(self, params: dict[str, Any]) -> list[float]:
        x0, x1 = params["x0"], params["x1"]
        f0 = x0
        f1 = (1.0 + x1) / x0
        return [f0, f1]

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        x0, x1 = params["x0"], params["x1"]
        c0 = 6.0 - 9.0 * x0 - x1
        c1 = 1.0 - 9.0 * x0 + x1
        return {"c0": c0, "c1": c1}


class ConstrainedBraninCurrin(BaseProblem):
    """Constrained Branin Currin Function, with the disk constraint
    from [Gelbart2014]_.

    The constraint's affine transform (``_con_bounds = [(-5, 10), (0, 15)]``,
    unnormalized) is numerically identical to the Branin rescale used by the
    objective (``15 * x - 5``, ``15 * x``), even though BoTorch defines them
    as two separate transforms.
    """

    _bounds = [(0.0, 1.0), (0.0, 1.0)]

    @property
    def search_space(self) -> dict[str, BaseDistribution]:
        return _box_search_space(self._bounds)

    @property
    def directions(self) -> list[StudyDirection]:
        return _MINIMIZE_2D

    @staticmethod
    def _rescale(x0: float, x1: float) -> tuple[float, float]:
        return 15.0 * x0 - 5.0, 15.0 * x1

    def evaluate(self, params: dict[str, Any]) -> list[float]:
        x0, x1 = np.float64(params["x0"]), np.float64(params["x1"])
        bx0, bx1 = self._rescale(x0, x1)
        t1 = bx1 - 5.1 / (4 * math.pi**2) * bx0**2 + 5 / math.pi * bx0 - 6
        t2 = 10 * (1 - 1 / (8 * math.pi)) * math.cos(bx0)
        branin = t1**2 + t2 + 10
        factor1 = 1 - math.exp(-1 / (2 * x1))
        numer = 2300 * x0**3 + 1900 * x0**2 + 2092 * x0 + 60
        denom = 100 * x0**3 + 500 * x0**2 + 4 * x0 + 20
        currin = factor1 * numer / denom
        return [branin, currin]

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        x0, x1 = params["x0"], params["x1"]
        bx0, bx1 = self._rescale(x0, x1)
        c0 = (bx0 - 2.5) ** 2 + (bx1 - 7.5) ** 2 - 50.0
        return {"c0": c0}


class OSY(BaseProblem):
    """The OSY test problem from [Oszycka1995]_, using the original
    minimization-form objectives verbatim (see module docstring)."""

    _bounds = [(0.0, 10.0), (0.0, 10.0), (1.0, 5.0), (0.0, 6.0), (1.0, 5.0), (0.0, 10.0)]

    @property
    def search_space(self) -> dict[str, BaseDistribution]:
        return _box_search_space(self._bounds)

    @property
    def directions(self) -> list[StudyDirection]:
        return _MINIMIZE_2D

    def evaluate(self, params: dict[str, Any]) -> list[float]:
        x0, x1, x2, x3, x4, x5 = (params[f"x{i}"] for i in range(6))
        f0 = -(25 * (x0 - 2) ** 2 + (x1 - 2) ** 2 + (x2 - 1) ** 2 + (x3 - 4) ** 2 + (x4 - 1) ** 2)
        f1 = x0**2 + x1**2 + x2**2 + x3**2 + x4**2 + x5**2
        return [f0, f1]

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        x0, x1, x2, x3, x4, x5 = (params[f"x{i}"] for i in range(6))
        c0 = 2.0 - x0 - x1
        c1 = x0 + x1 - 6.0
        c2 = x1 - x0 - 2.0
        c3 = x0 - 3.0 * x1 - 2.0
        c4 = (x2 - 3.0) ** 2 + x3 - 4.0
        c5 = 4.0 - (x4 - 3.0) ** 2 - x5
        return {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


class SRN(BaseProblem):
    """The constrained SRN problem. See [GarridoMerchan2020]_. Minimization."""

    _bounds = [(-20.0, 20.0), (-20.0, 20.0)]

    @property
    def search_space(self) -> dict[str, BaseDistribution]:
        return _box_search_space(self._bounds)

    @property
    def directions(self) -> list[StudyDirection]:
        return _MINIMIZE_2D

    def evaluate(self, params: dict[str, Any]) -> list[float]:
        x0, x1 = params["x0"], params["x1"]
        f0 = 2.0 + (x0 - 2.0) ** 2 + (x1 - 2.0) ** 2
        f1 = 9.0 * x0 - (x1 - 1.0) ** 2
        return [f0, f1]

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        x0, x1 = params["x0"], params["x1"]
        c0 = x0**4 + x1**4 - 225.0
        c1 = 10.0 + x0 - 3.0 * x1
        return {"c0": c0, "c1": c1}


class MW7(BaseProblem):
    """The MW7 problem: 2 objectives, 2 constraints, disconnected Pareto front.

    Supports arbitrary input dimension > 1. See [Ma2019]_ for details.
    """

    def __init__(self, dim: int) -> None:
        if dim < 2:
            raise ValueError("dim must be greater than or equal to 2.")
        self.dim = dim

    @property
    def search_space(self) -> dict[str, BaseDistribution]:
        return {f"x{i}": FloatDistribution(low=0.0, high=1.0) for i in range(self.dim)}

    @property
    def directions(self) -> list[StudyDirection]:
        return _MINIMIZE_2D

    def evaluate(self, params: dict[str, Any]) -> list[float]:
        xs = [np.float64(params[f"x{i}"]) for i in range(self.dim)]
        contrib = sum(2 * (xs[i + 1] + (xs[i] - 0.5) ** 2 - 1) ** 2 for i in range(self.dim - 1))
        g = 1 + contrib
        f0 = g * xs[0]
        f1 = g * math.sqrt(1 - (f0 / g) ** 2)
        return [f0, f1]

    def evaluate_constraints(self, params: dict[str, Any]) -> dict[str, float]:
        def la2(a: float, b: float, c: float, d: float, theta: float) -> float:
            return a * math.sin(b * theta**c) ** d

        f0, f1 = self.evaluate(params)
        atan = math.atan(f1 / f0)
        c0 = f0**2 + f1**2 - (1.2 + abs(la2(0.4, 4.0, 1.0, 16.0, atan))) ** 2
        c1 = (1.15 - la2(0.2, 4.0, 1.0, 8.0, atan)) ** 2 - f0**2 - f1**2
        return {"c0": c0, "c1": c1}
