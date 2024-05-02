"""
Joint/grouped score of Support and Sufficiency.
"""
from typing import List, Dict, Union
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from metrics.group import GroupMetric
from metrics.support import SupportMetric


@dataclass
class GoldPredictionInstance:
    gold_supporting_facts: List = field(default_factory=lambda: deepcopy([]))
    predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))

    gold_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class GroupSupportSufficiencyMetric(GroupMetric):
    def __init__(self) -> None:
        self.prediction_store = defaultdict(GoldPredictionInstance)
        self.support_metric = SupportMetric()

    def compute_question_scores(
        self, group: GoldPredictionInstance
    ) -> Dict[str, float]:

        # Call it only when reset=True
        assert group.gold_supporting_facts is not None
        assert group.predicted_supporting_facts is not None
        assert len(group.predicted_sufficiencies) == 2

        self.support_metric(
            group.predicted_supporting_facts, group.gold_supporting_facts
        )
        sp_em, sp_f1 = self.support_metric.get_metric(reset=True)

        sufficiency_score = group.predicted_sufficiencies == group.gold_sufficiencies
        sp_f1 = sp_f1 if sufficiency_score else 0.0
        sp_em = sp_em if sufficiency_score else 0.0
        sufficiency_score = float(sufficiency_score)

        question_scores = {"f1": sp_f1, "em": sp_em, "suff": sufficiency_score}
        return question_scores

    def __call__(
        self,
        predicted_supporting_facts: List,
        gold_supporting_facts: List,
        predicted_sufficiency: int,
        gold_sufficiency: int,
        question_id: Union[int, str],
    ) -> None:

        question_id = str(question_id)

        if gold_sufficiency == 1:
            self.prediction_store[
                question_id
            ].gold_supporting_facts = gold_supporting_facts
            self.prediction_store[
                question_id
            ].predicted_supporting_facts = predicted_supporting_facts

        self.prediction_store[question_id].predicted_sufficiencies.append(
            predicted_sufficiency
        )
        self.prediction_store[question_id].gold_sufficiencies.append(gold_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(GoldPredictionInstance)
