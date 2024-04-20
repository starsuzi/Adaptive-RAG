"""
Support metric -- mostly taken directly from hotpotqa
"""
from typing import Tuple, List, Dict
import ftfy
import re

from metrics.metric import Metric
from metrics.squad_answer_em_f1 import normalize_answer


def compute_metrics(predicted_support: List[str], gold_support: List[str]) -> Dict:
    # Taken from hotpot_eval

    predicted_support = set([re.sub(r" +", "", ftfy.fix_text(str(e)).lower()) for e in predicted_support])
    gold_support = set([re.sub(r" +", "", ftfy.fix_text(str(e)).lower()) for e in gold_support])

    tp, fp, fn = 0, 0, 0
    for e in predicted_support:
        if e in gold_support:
            tp += 1
        else:
            fp += 1
    for e in gold_support:
        if e not in predicted_support:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    # In case everything is empty, set both f1, em to be 1.0.
    # Without this change, em gets 1 and f1 gets 0
    if not predicted_support and not gold_support:
        f1, em = 1.0, 1.0
        f1, em = 1.0, 1.0

    return {"prec": prec, "recall": recall, "f1": f1, "em": em}


class SupportEmF1Metric(Metric):
    """
    SupportMetric: Em and F1 (Similar to HotpotQA Sp metric)
    """

    def __init__(self, do_normalize_answer: bool = False) -> None:
        self._titles_total_em = 0.0
        self._titles_total_f1 = 0.0
        self._titles_total_precision = 0.0
        self._titles_total_recall = 0.0

        self._paras_total_em = 0.0
        self._paras_total_f1 = 0.0
        self._paras_total_precision = 0.0
        self._paras_total_recall = 0.0

        self._total_predicted_titles = 0
        self._max_predicted_titles = -float("inf")
        self._min_predicted_titles = float("inf")

        self._total_predicted_paras = 0
        self._max_predicted_paras = -float("inf")
        self._min_predicted_paras = float("inf")

        self._do_normalize_answer = do_normalize_answer
        self._count = 0

    def __call__(self, predicted_support: List[str], gold_support: List[str]):

        predicted_support = predicted_support or []

        if self._do_normalize_answer:
            predicted_support = [normalize_answer(e) for e in predicted_support]
            gold_support = [normalize_answer(e) for e in gold_support]

        if not gold_support:
            gold_support_titles = []
            gold_support_paras = []
            predicted_support_titles = predicted_support_paras = predicted_support

        elif gold_support[0].startswith("pid"):
            for e in gold_support + predicted_support:
                assert e.startswith("pid")
            predicted_support_titles = [e.split("___")[1] for e in predicted_support]
            predicted_support_paras = predicted_support
            gold_support_titles = [e.split("___")[1] for e in gold_support]
            gold_support_paras = gold_support

        else:
            for e in gold_support + predicted_support:
                assert not e.startswith("pid")
            predicted_support_titles = predicted_support_paras = predicted_support
            gold_support_titles = gold_support_paras = gold_support

        predicted_support_titles = set(map(str, predicted_support_titles))
        predicted_support_paras = set(map(str, predicted_support_paras))

        gold_support_titles = set(map(str, gold_support_titles))
        gold_support_paras = set(map(str, gold_support_paras))

        titles_metrics = compute_metrics(predicted_support_titles, gold_support_titles)
        paras_metrics = compute_metrics(predicted_support_paras, gold_support_paras)

        self._total_predicted_titles += len(predicted_support_titles)
        self._max_predicted_titles = max(self._max_predicted_titles, len(predicted_support_titles))
        self._min_predicted_titles = min(self._min_predicted_titles, len(predicted_support_titles))

        self._total_predicted_paras += len(predicted_support_paras)
        self._max_predicted_paras = max(self._max_predicted_titles, len(predicted_support_paras))
        self._min_predicted_paras = min(self._min_predicted_titles, len(predicted_support_paras))

        self._titles_total_em += float(titles_metrics["em"])
        self._titles_total_f1 += titles_metrics["f1"]
        self._titles_total_precision += titles_metrics["prec"]
        self._titles_total_recall += titles_metrics["recall"]

        self._paras_total_em += float(paras_metrics["em"])
        self._paras_total_f1 += paras_metrics["f1"]
        self._paras_total_precision += paras_metrics["prec"]
        self._paras_total_recall += paras_metrics["recall"]

        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order).
        """
        titles_exact_match = self._titles_total_em / self._count if self._count > 0 else 0
        titles_f1_score = self._titles_total_f1 / self._count if self._count > 0 else 0
        titles_precision_score = self._titles_total_precision / self._count if self._count > 0 else 0
        titles_recall_score = self._titles_total_recall / self._count if self._count > 0 else 0

        paras_exact_match = self._paras_total_em / self._count if self._count > 0 else 0
        paras_f1_score = self._paras_total_f1 / self._count if self._count > 0 else 0
        paras_precision_score = self._paras_total_precision / self._count if self._count > 0 else 0
        paras_recall_score = self._paras_total_recall / self._count if self._count > 0 else 0

        avg_predicted_titles = self._total_predicted_titles / self._count if self._count > 0 else 0
        avg_predicted_paras = self._total_predicted_paras / self._count if self._count > 0 else 0

        if reset:
            self.reset()

        return {
            "title_em": round(titles_exact_match, 3),
            "title_f1": round(titles_f1_score, 3),
            "title_precision": round(titles_precision_score, 3),
            "title_recall": round(titles_recall_score, 3),
            "para_em": round(paras_exact_match, 3),
            "para_f1": round(paras_f1_score, 3),
            "para_precision": round(paras_precision_score, 3),
            "para_recall": round(paras_recall_score, 3),
            "avg_predicted_titles": avg_predicted_titles,
            "max_predicted_titles": self._max_predicted_titles,
            "min_predicted_titles": self._min_predicted_titles,
            "avg_predicted_paras": avg_predicted_paras,
            "max_predicted_paras": self._max_predicted_paras,
            "min_predicted_paras": self._min_predicted_paras,
            "count": self._count,
        }

    def reset(self):

        self._titles_total_em = 0.0
        self._titles_total_f1 = 0.0
        self._titles_total_precision = 0.0
        self._titles_total_recall = 0.0

        self._paras_total_em = 0.0
        self._paras_total_f1 = 0.0
        self._paras_total_precision = 0.0
        self._paras_total_recall = 0.0

        self._total_predicted_titles = 0
        self._max_predicted_titles = -float("inf")
        self._min_predicted_titles = float("inf")

        self._total_predicted_paras = 0
        self._max_predicted_paras = -float("inf")
        self._min_predicted_paras = float("inf")

        self._count = 0
