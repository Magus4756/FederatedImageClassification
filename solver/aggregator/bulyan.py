# encode=utf-8

from .krum import Krum
from .trimmed_mean import TrimmedMean


class Bulyan:
    """

    """
    def __init__(self, num_compromised: int):
        """

        :param num_compromised: 恶意参与方数量
        """
        self._f = num_compromised

    def __call__(self, weights: list):
        assert len(weights) >= 4 * self._f + 3
        chosen = []
        n = len(weights) - 2 * self._f
        for _ in range(n):
            chosen_w, idx = Krum(self._f).krum(weights)
            chosen.append(chosen_w)
            weights.pop(idx)
        return TrimmedMean(2 * self._f)(chosen)


