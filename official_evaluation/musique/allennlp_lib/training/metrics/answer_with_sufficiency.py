from typing import List, Dict, Union
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy

from overrides import overrides

from allennlp_lib.training.metrics.group import Group
from allennlp_lib.training.metrics import SquadEmAndF1


@dataclass
class LabelPredictionInstance:
    label_answers: str = None
    predicted_answer: str = None

    label_sufficiencies: List = field(default_factory=lambda: deepcopy([]))
    predicted_sufficiencies: List = field(default_factory=lambda: deepcopy([]))


class AnswerWithSufficiency(Group):

    def __init__(self) -> None:
        self.prediction_store = defaultdict(LabelPredictionInstance)
        # self.score_store = defaultdict(dict)
        self.answer_metric = SquadEmAndF1()

    @overrides
    def compute_question_scores(self, group: LabelPredictionInstance) -> Dict[str, float]:

        # if (group.label_answers is None or group.predicted_answer is None
        #     or len(group.predicted_sufficiencies) != 2):
        #     return {"f1": 0.0, "em": 0.0, "precision": 0.0, "recall": 0.0}

        # Call it only when reset=True
        assert group.label_answers is not None
        assert group.predicted_answer is not None
        assert len(group.predicted_sufficiencies) == 2

        assert isinstance(group.label_answers, list)
        self.answer_metric(group.predicted_answer, group.label_answers)
        ans_em, ans_f1 = self.answer_metric.get_metric(reset=True)

        sufficiency_score = group.predicted_sufficiencies == group.label_sufficiencies
        ans_f1 = ans_f1 if sufficiency_score else 0.0
        ans_em = ans_em if sufficiency_score else 0.0
        sufficiency_score = float(sufficiency_score)

        question_scores = {"f1": ans_f1, "em": ans_em, "suff": sufficiency_score}
        return question_scores

    @overrides
    def __call__(self,
                 predicted_answer: str,
                 label_answers: str,
                 predicted_sufficiency: int,
                 label_sufficiency: int,
                 question_id: Union[int, str],
                 ) -> None:

        question_id = str(question_id)
        predicted_sufficiency, label_sufficiency = self.detach_tensors(predicted_sufficiency, label_sufficiency)

        if label_sufficiency == 1:
            self.prediction_store[question_id].predicted_answer = predicted_answer
            self.prediction_store[question_id].label_answers = label_answers

        self.prediction_store[question_id].predicted_sufficiencies.append(predicted_sufficiency)
        self.prediction_store[question_id].label_sufficiencies.append(label_sufficiency)

    def reset(self):
        self.prediction_store = defaultdict(LabelPredictionInstance)
        # self.score_store = defaultdict(dict)
