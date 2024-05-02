# Mostly taken from HotpotQA evaluation script.

import sys
import json
import re
import string
from collections import Counter
from typing import List, Tuple, Union


def normalize_answer(s: str):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def ans_f1(prediction: str, ground_truth: str):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def ans_em(prediction: str, ground_truth: str):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def sp_f1(prediction: List[Union[Tuple, str]], gold: List[Union[Tuple, str]]):

    cur_sp_pred = {e if isinstance(e, str) else tuple(e) for e in prediction}
    gold_sp_pred = {e if isinstance(e, str) else tuple(e) for e in gold}

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
    return f1, prec, recall


def sp_em(prediction: List[Union[Tuple, str]], gold: List[Union[Tuple, str]]):
    cur_sp_pred = {e if isinstance(e, str) else tuple(e) for e in prediction}
    gold_sp_pred = {e if isinstance(e, str) else tuple(e) for e in gold}

    return float(cur_sp_pred == gold_sp_pred) 
