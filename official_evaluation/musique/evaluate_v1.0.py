import json
import argparse
from typing import List, Dict
from collections import Counter

from metrics.answer import AnswerMetric
from metrics.support import SupportMetric
from metrics.group_answer_sufficiency import GroupAnswerSufficiencyMetric
from metrics.group_support_sufficiency import GroupSupportSufficiencyMetric


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def evaluate(filepath_with_predictions: str, filepath_with_ground_truths: str) -> Dict:

    prediction_instances = read_jsonl(filepath_with_predictions)
    ground_truth_instances = read_jsonl(filepath_with_ground_truths)

    do_sufficiency_eval = False
    answer_metric = AnswerMetric()
    support_metric = SupportMetric()
    group_answer_sufficiency_metric = GroupAnswerSufficiencyMetric()
    group_support_sufficiency_metric = GroupSupportSufficiencyMetric()

    assert len(prediction_instances) == len(
        ground_truth_instances
    ), "The number of lines in the two files are not the same."

    for ground_truth_instance, prediction_instance in zip(
        ground_truth_instances, prediction_instances
    ):

        assert (
            ground_truth_instance["id"] == prediction_instance["id"]
        ), "The instances (ids) in prediction and gold filepath jsonl should be in same order."

        question_id = ground_truth_instance["id"]

        predicted_answer = prediction_instance["predicted_answer"]
        ground_truth_answers = [
            ground_truth_instance["answer"]
        ] + ground_truth_instance["answer_aliases"]

        predicted_support_indices = prediction_instance["predicted_support_idxs"]
        ground_truth_support_indices = [
            paragraph["idx"]
            for paragraph in ground_truth_instance["paragraphs"]
            if paragraph["is_supporting"]
        ]

        predicted_sufficiency = prediction_instance["predicted_answerable"]
        ground_truth_sufficiency = ground_truth_instance["answerable"]

        if ground_truth_sufficiency:
            answer_metric(predicted_answer, ground_truth_answers)
            support_metric(predicted_support_indices, ground_truth_support_indices)

        group_answer_sufficiency_metric(
            predicted_answer,
            ground_truth_answers,
            predicted_sufficiency,
            ground_truth_sufficiency,
            question_id,
        )
        group_support_sufficiency_metric(
            predicted_support_indices,
            ground_truth_support_indices,
            predicted_sufficiency,
            ground_truth_sufficiency,
            question_id,
        )

        # If there's any instance with ground truth of unanswerable, we'll assume
        # it's full version of the dataset and not only the answerable version.
        if not ground_truth_sufficiency:
            do_sufficiency_eval = True

    metrics = {}
    metrics["answer_f1"] = round(answer_metric.get_metric()[1], 3)
    metrics["answer_em"] = round(answer_metric.get_metric()[0], 3)
    metrics["answer_acc"] = round(answer_metric.get_metric()[2], 3)
    metrics["support_f1"] = round(support_metric.get_metric()[1], 3)

    if do_sufficiency_eval:
        assert set(Counter([e['id'] for e in prediction_instances]).values()) == {2}, \
            "For sufficiency evaluation, there should two instances for each question."

        metrics["group_answer_sufficiency_f1"] = round(
            group_answer_sufficiency_metric.get_metric()["f1"], 3
        )
        metrics["group_support_sufficiency_f1"] = round(
            group_support_sufficiency_metric.get_metric()["f1"], 3
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MuSiQue predictions.")
    parser.add_argument(
        "filepath_with_predictions",
        type=str,
        help="jsonl filepath to predicted instances.",
    )
    parser.add_argument(
        "filepath_with_ground_truths",
        type=str,
        help="jsonl filepath to data instances.",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        help="(optional) filepath to save output metrics."
    )
    args = parser.parse_args()

    metrics = evaluate(args.filepath_with_predictions, args.filepath_with_ground_truths)

    if args.output_filepath:
        print(f"Writing metrics output in: {args.output_filepath}")
        with open(args.output_filepath, "w") as file:
            json.dump(metrics, file, indent=4)
    else:
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
