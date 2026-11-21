import numpy as np
import pytest

from anomaly_detector.detectors.base import BaseDetector
from anomaly_detector.ensemble import EnsembleDetector


class Fixed(BaseDetector):
    name = "fixed"

    def __init__(self, flags):
        super().__init__(threshold=0.5)
        self.flags = np.asarray(flags, dtype=float)

    def fit(self, data):
        self._fitted = True
        return self

    def score(self, data):
        return self.flags


@pytest.fixture
def three():
    return [Fixed([1, 1, 0, 0]), Fixed([1, 0, 1, 0]), Fixed([1, 0, 0, 0])]


@pytest.mark.parametrize(
    "strategy, expected",
    [("any", [1, 1, 1, 0]), ("all", [1, 0, 0, 0]), ("majority", [1, 0, 0, 0])],
)
def test_strategies(three, strategy, expected):
    data = np.zeros(4)
    out = EnsembleDetector(three, strategy=strategy).fit_predict(data)
    assert out.tolist() == [bool(x) for x in expected]


def test_majority_with_two_of_three():
    dets = [Fixed([1, 0]), Fixed([1, 1]), Fixed([0, 1])]
    assert EnsembleDetector(dets, "majority").fit_predict(np.zeros(2)).tolist() == [True, True]


def test_votes_shape(three):
    votes = EnsembleDetector(three).fit(np.zeros(4)).votes(np.zeros(4))
    assert votes.shape == (4, 3)


def test_validation():
    with pytest.raises(ValueError):
        EnsembleDetector([])
    with pytest.raises(ValueError):
        EnsembleDetector([Fixed([0])], strategy="sometimes")
