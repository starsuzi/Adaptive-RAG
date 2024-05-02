"""
An abstract class representing a metric which can be accumulated.
"""
from typing import Any, Dict


class Metric:
    """
    An abstract class representing a metric which can be accumulated.
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError
