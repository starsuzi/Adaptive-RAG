"""
Answer support recall as a measure of retrieval performance.
"""
from typing import Tuple, List
import re

from metrics.metric import Metric
from metrics.squad_answer_em_f1 import normalize_answer


class AnswerSupportRecallMetric(Metric):
    """
    AnswerSupportRecall: Recall of the presense of the answer/s in the retrieved paras.
    """

    def __init__(self) -> None:
        self._total_count = 0
        self._total_num_retrieved_paras = 0
        self._total_answer_support_recall = 0

    def __call__(self, predicted_paragraph_texts: List[str], gold_answers: List[str]):

        answer_covered_count = 0
        for gold_answer in gold_answers:
            for predicted_paragraph_text in predicted_paragraph_texts:

                def lower_clean_ws(e):
                    return re.sub(" +", " ", e.lower().strip())

                condition_1 = lower_clean_ws(gold_answer) in lower_clean_ws(predicted_paragraph_text)
                condition_2 = normalize_answer(gold_answer) in normalize_answer(predicted_paragraph_text)
                if condition_1 or condition_2:
                    answer_covered_count += 1
                    break

        answer_support_recall = answer_covered_count / len(gold_answers)
        self._total_answer_support_recall += answer_support_recall
        self._total_num_retrieved_paras += len(predicted_paragraph_texts)
        self._total_count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average answer occurrence recall and number of paragraphs.
        """

        avg_answer_support_recall = (
            self._total_answer_support_recall / self._total_count if self._total_count > 0 else 0
        )
        avg_retrieved_paras = self._total_num_retrieved_paras / self._total_count if self._total_count > 0 else 0

        avg_answer_support_recall = round(avg_answer_support_recall, 3)
        avg_retrieved_paras = round(avg_retrieved_paras, 3)

        if reset:
            self.reset()

        return {
            "answer_support_recall": avg_answer_support_recall,
            "avg_predicted_paras": avg_retrieved_paras,
            "count": self._total_count,
        }

    def reset(self):
        self._total_count = 0
        self._total_num_retrieved_paras = 0
        self._total_answer_support_recall = 0
