"""Parity tests for the ``botorch_cmo_test_funcs`` OptunaHub package.

Compares against ``golden.json`` -- ``evaluate_true``/``evaluate_slack_true`` values
recorded from real BoTorch for the same 6 problems. Per BoTorch's ``>=0``-feasible
slack vs. OptunaHub's ``<=0``-feasible constraint convention, ``evaluate_constraints``
is checked against the *negated* recorded slacks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from typing import TYPE_CHECKING

from golden import GOLDEN
from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
import optunahub
import pytest


if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet


ob = optunahub.load_local_module(
    package="benchmarks/botorch_cmo_test_funcs", registry_root="package"
)

RTOL = 1e-9
ATOL = 1e-9


class _ConstrainedProblem(Protocol):
    """Structural type for the ``BaseProblem`` subclasses under test.

    ``BaseProblem`` itself declares neither ``evaluate_constraints`` nor a
    ``Sequence[float]``-typed ``evaluate``/``FloatDistribution``-valued
    ``search_space``; every problem in ``SPECS`` provides all three.
    """

    directions: list[StudyDirection]
    search_space: dict[str, FloatDistribution]

    def evaluate(self, params: dict[str, float]) -> Sequence[float]: ...

    def evaluate_constraints(self, params: dict[str, float]) -> dict[str, float]: ...


# ``cls`` here is the OptunaHub wrapper class name, which matches the BoTorch
# class name in ``golden.json`` for every problem except ``MW7``, which is
# recorded there at two dimensions under one shared class.
SPECS = [
    ("BNH", "BNH", {}),
    ("CONSTR", "CONSTR", {}),
    ("ConstrainedBraninCurrin", "ConstrainedBraninCurrin", {}),
    ("OSY", "OSY", {}),
    ("SRN", "SRN", {}),
    ("MW7(dim=2)", "MW7", {"dim": 2}),
    ("MW7(dim=5)", "MW7", {"dim": 5}),
]


def build(cls_name: str, init_kwargs: dict) -> _ConstrainedProblem:
    return getattr(ob, cls_name)(**init_kwargs)


def as_params(X: list[float]) -> dict[str, float]:
    return {f"x{i}": value for i, value in enumerate(X)}


def case_ids(golden_key: str) -> list[ParameterSet]:
    return [
        pytest.param(golden_key, index, id=f"{golden_key}-{case['name']}")
        for index, case in enumerate(GOLDEN[golden_key]["cases"])
    ]


ALL_CASES = [param for golden_key, _, _ in SPECS for param in case_ids(golden_key)]
SPEC_BY_GOLDEN_KEY = {
    golden_key: (cls_name, init_kwargs) for golden_key, cls_name, init_kwargs in SPECS
}


@pytest.mark.parametrize("golden_key, index", ALL_CASES)
def test_evaluate_matches_recorded_botorch_objectives(golden_key: str, index: int) -> None:
    cls_name, init_kwargs = SPEC_BY_GOLDEN_KEY[golden_key]
    case = GOLDEN[golden_key]["cases"][index]
    actual = build(cls_name, init_kwargs).evaluate(as_params(case["X"]))
    for got, expected in zip(actual, case["objectives"], strict=True):
        assert got == pytest.approx(expected, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("golden_key, index", ALL_CASES)
def test_evaluate_constraints_matches_negated_recorded_botorch_slacks(
    golden_key: str, index: int
) -> None:
    cls_name, init_kwargs = SPEC_BY_GOLDEN_KEY[golden_key]
    case = GOLDEN[golden_key]["cases"][index]
    actual = build(cls_name, init_kwargs).evaluate_constraints(as_params(case["X"]))
    expected = [-slack for slack in case["slacks"]]
    for got, exp in zip(actual.values(), expected, strict=True):
        assert got == pytest.approx(exp, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("golden_key, index", ALL_CASES)
def test_directions_are_all_minimize(golden_key: str, index: int) -> None:
    del index
    cls_name, init_kwargs = SPEC_BY_GOLDEN_KEY[golden_key]
    problem = build(cls_name, init_kwargs)
    assert problem.directions == [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]


@pytest.mark.parametrize("golden_key", [k for k, _, _ in SPECS])
def test_search_space_matches_recorded_botorch_bounds(golden_key: str) -> None:
    cls_name, init_kwargs = SPEC_BY_GOLDEN_KEY[golden_key]
    entry = GOLDEN[golden_key]
    problem = build(cls_name, init_kwargs)
    space = problem.search_space
    assert list(space) == [f"x{i}" for i in range(entry["dim"])]
    lower, upper = entry["bounds"]
    for i, dist in enumerate(space.values()):
        assert dist.low == pytest.approx(lower[i])
        assert dist.high == pytest.approx(upper[i])


def test_mw7_rejects_dim_below_two() -> None:
    with pytest.raises(ValueError):
        ob.MW7(dim=1)
