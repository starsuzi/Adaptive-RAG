from typing import Dict
import argparse
import json
import os

from utils.common import read_jsonl, write_jsonl, translate_id


def raw_prediction_to_official_format(instance: Dict) -> Dict:
    """
    Prediction Format. It should be jsonl in the same order as the input.
    {
        "id": <id>,
        "predicted_support_idxs": [...],
        "predicted_answer": "...",
        "predicted_answerable": True/False
    }
    """

    if "predicted_ordered_contexts" in instance["input"]:
        # means predicted supported indices are indices in predicted_ordered_contexts
        # and not ordered_contexts.

        predicted_select_support_indices = [
            instance['input']['contexts'].index(instance['input']['predicted_ordered_contexts'][index])
            for index in instance["predicted_select_support_indices"]
        ]
    else:
        predicted_select_support_indices = instance["predicted_select_support_indices"]


    translated_instance = {
        "id": translate_id(instance["input"]["id"]),
        "predicted_answer": instance["predicted_best_answer"],
        "predicted_support_idxs": predicted_select_support_indices,
        "predicted_answerable": bool(instance.get("predicted_answerability", False))
    }
    return translated_instance

def raw_predictions_to_official_format(
        source_filepath: str,
        target_filepath: str
    ):

    source_instances = read_jsonl(source_filepath)

    translated_instances = []
    for instance in source_instances:
        translated_instance = raw_prediction_to_official_format(instance)
        translated_instances.append(translated_instance)

    write_jsonl(translated_instances, target_filepath)


def main():

    parser = argparse.ArgumentParser(description='Convert raw predictions file to official format.')
    parser.add_argument(
        'input_filepath',
        type=str,
        help='filepath to raw predictions file.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_filepath):
        exit(f"Filepath {args.input_filepath} not found.")

    output_filepath = "_official_format".join(os.path.splitext(args.input_filepath))
    raw_predictions_to_official_format(args.input_filepath, output_filepath)
    

if __name__ == '__main__':
    main()
