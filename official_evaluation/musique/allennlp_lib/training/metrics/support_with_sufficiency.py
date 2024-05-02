from typing import List, Dict, Union
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from overrides import overrides

from allennlp_lib.training.metrics.group import Group
from allennlp_lib.training.metrics import ListCompareEmAndF1


@dataclass
class LabelPredictionInstance:
    label_supporting_facts: List = field(default_factory=lambda: deepcopy([]))
    predicted_supporting_facts: List = field(default_factory=lambda: deepcopy([]))

    label_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class SupportWithSufficiency(Group):

    def __init__(self) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        # self.score_store = defaultdict(dict)
        self.support_metric = ListCompareEmAndF1()

    @overrides
    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        # if (group.label_supporting_facts is None or group.predicted_supporting_facts is None
        #     or len(group.predicted_sufficiencies) != 2):
        #     return {"f1": 0.0, "em": 0.0, "precision": 0.0, "recall": 0.0}

        # Call it only when reset=True
        assert group.label_supporting_facts is not None
        assert group.predicted_supporting_facts is not None
        assert len(group.predicted_sufficiencies) == 2

        self.support_metric(group.predicted_supporting_facts, group.label_supporting_facts)
        sp_em, sp_f1 = self.support_metric.get_metric(reset=True)

        sufficiency_score = group.predicted_sufficiencies == group.label_sufficiencies
        sp_f1 = sp_f1 if sufficiency_score else 0.0
        sp_em = sp_em if sufficiency_score else 0.0
        sufficiency_score = float(sufficiency_score)

        question_scores = {"f1": sp_f1, "em": sp_em, "suff": sufficiency_score}
        return question_scores

    @overrides
    def __call__(self,
                 predicted_supporting_facts: List,
                 label_supporting_facts: List,
                 predicted_sufficiency: int,
                 label_sufficiency: int,
                 question_id: Union[int, str],
                 ) -> None:

        question_id = str(question_id)
        predicted_sufficiency, label_sufficiency = self.detach_tensors(predicted_sufficiency, label_sufficiency)

        if label_sufficiency == 1:
            self.prediction_store[question_id].label_supporting_facts = label_supporting_facts
            self.prediction_store[question_id].predicted_supporting_facts = predicted_supporting_facts

        self.prediction_store[question_id].predicted_sufficiencies.append(predicted_sufficiency)
        self.prediction_store[question_id].label_sufficiencies.append(label_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        # self.score_store = defaultdict(dict)
