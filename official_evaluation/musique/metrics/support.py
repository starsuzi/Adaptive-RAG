"""
Support metric -- mostly taken directly from hotpotqa
"""
from typing import Tuple, List

from metrics.metric import Metric


class SupportMetric(Metric):
    """
    SupportMetric: Em and F1 (Similar to HotpotQA Sp metric)
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._count = 0

    def __call__(self, predicted_support_idxs: List[int], gold_support_idxs: List[int]):

        # Taken from hotpot_eval
        cur_sp_pred = set(map(int, predicted_support_idxs))
        gold_sp_pred = set(map(int, gold_support_idxs))
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0

        # In case everything is empty, set both f1, em to be 1.0.
        # Without this change, em gets 1 and f1 gets 0
        if not cur_sp_pred and not gold_sp_pred:
            f1, em = 1.0, 1.0
            f1, em = 1.0, 1.0

        self._total_em += float(em)
        self._total_f1 += f1
        self._total_precision += prec
        self._total_recall += recall
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order).
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        # precision_score = self._total_precision / self._count if self._count > 0 else 0
        # recall_score = self._total_recall / self._count if self._count > 0 else 0

        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._count = 0
