"""
Joint/grouped score of Answer and Sufficiency.
"""
from typing import List, Dict, Union
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics.group import GroupMetric
from metrics.answer import AnswerMetric


@dataclass
class GoldPredictionInstance:
    gold_answers: str = None
    predicted_answer: str = None

    gold_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class GroupAnswerSufficiencyMetric(GroupMetric):
    def __init__(self) -> None:
        self.prediction_store = defaultdict(GoldPredictionInstance)
        self.answer_metric = AnswerMetric()

    def compute_question_scores(
        self, group: GoldPredictionInstance
    ) -> Dict[str, float]:

        # Call it only when reset=True
        assert group.gold_answers is not None
        assert group.predicted_answer is not None
        assert len(group.predicted_sufficiencies) == 2

        assert isinstance(group.gold_answers, list)
        self.answer_metric(group.predicted_answer, group.gold_answers)
        ans_em, ans_f1 = self.answer_metric.get_metric(reset=True)

        sufficiency_score = group.predicted_sufficiencies == group.gold_sufficiencies
        ans_f1 = ans_f1 if sufficiency_score else 0.0
        ans_em = ans_em if sufficiency_score else 0.0
        sufficiency_score = float(sufficiency_score)

        question_scores = {"f1": ans_f1, "em": ans_em, "suff": sufficiency_score}
        return question_scores

    def __call__(
        self,
        predicted_answer: str,
        gold_answers: str,
        predicted_sufficiency: int,
        gold_sufficiency: int,
        question_id: Union[int, str],
    ) -> None:

        question_id = str(question_id)

        if gold_sufficiency == 1:
            self.prediction_store[question_id].predicted_answer = predicted_answer
            self.prediction_store[question_id].gold_answers = gold_answers

        self.prediction_store[question_id].predicted_sufficiencies.append(
            predicted_sufficiency
        )
        self.prediction_store[question_id].gold_sufficiencies.append(gold_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(GoldPredictionInstance)
