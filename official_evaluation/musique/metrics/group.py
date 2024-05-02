"""
Abstract class to group metrics together.
"""
from typing import Dict

from metrics.metric import Metric


class GroupMetric(Metric):
    """
    Abstract class to group metrics together.
    """

    def __init__(self) -> None:
        self.reset()

    def compute_question_scores(self, group) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        total_scores = {"f1": 0.0, "em": 0.0, "suff": 0.0}
        for question_id, question_group in self.prediction_store.items():
            question_scores = self.compute_question_scores(question_group)
            # self.score_store[question_id] = question_scores
            for key, value in question_scores.items():
                total_scores[key] += value
        dataset_scores = {
            name: total_score / len(self.prediction_store)
            if len(self.prediction_store) > 0
            else 0.0
            for name, total_score in total_scores.items()
        }

        if reset:
            self.reset()

        return dataset_scores
