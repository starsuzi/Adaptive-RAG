from typing import Tuple, List

import ftfy
from metrics.metric import Metric
from metrics.drop_eval import (
    get_metrics as drop_em_and_f1,
)
from metrics.squad_answer_em_f1 import metric_max_over_ground_truths


class DropAnswerEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_prec = 0.0
        self._total_recall = 0.0
        self._count = 0

    def __call__(
        self,
        predicted_answer_list: List[str],
        list_of_ground_truth_answer_list: List[List[str]],
    ):
        assert isinstance(predicted_answer_list, (list, tuple))
        assert isinstance(list_of_ground_truth_answer_list, (list, tuple))

        if not predicted_answer_list:
            predicted_answer_list = [""]

        assert isinstance(predicted_answer_list[0], str)
        assert isinstance(list_of_ground_truth_answer_list[0], (list, tuple))
        assert isinstance(list_of_ground_truth_answer_list[0][0], str)

        predicted_answer_list = [ftfy.fix_text(e) for e in predicted_answer_list]
        list_of_ground_truth_answer_list = [
            [ftfy.fix_text(e) for e in ground_truth_answer_list]
            for ground_truth_answer_list in list_of_ground_truth_answer_list
        ]

        exact_match, f1_score, prec_score, recall_score = metric_max_over_ground_truths(
            drop_em_and_f1, predicted_answer_list, list_of_ground_truth_answer_list
        )

        # Converting to int here, since we want to count the number of exact matches.
        self._total_em += int(exact_match)
        self._total_f1 += f1_score
        self._total_prec += prec_score
        self._total_recall += recall_score
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        prec_score = self._total_prec / self._count if self._count > 0 else 0
        recall_score = self._total_recall / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {
            "em": round(exact_match, 3),
            "f1": round(f1_score, 3),
            "precision": round(prec_score, 3),
            "recall": round(recall_score, 3),
            "count": self._count,
        }

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_prec = 0.0
        self._total_recall = 0.0
        self._count = 0
