import re
import os
import json, jsonlines
import uuid
import subprocess
import argparse
from typing import Dict, Any
import string

import _jsonnet
from lib import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    get_config_file_path_from_name_or_path,
)
from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from metrics.support_em_f1 import SupportEmF1Metric
from metrics.answer_support_recall import AnswerSupportRecallMetric
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric

# Set your path accordingly
base_pred_path = './predictions/classifier/t5-large/flan_t5_xl/epoch/25/2024_04_19/01_53_50/'

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))) 

def answer_extractor(potentially_cot: str) -> str:
    # In a few experiments I forgot the configuring the answer extractor part
    # and so the final answer is a cot chain instead. Instead of having to do
    # all those exps again, I'm just doing answer_extraction here. This needs
    # to be fixed later though.

    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output

def load_ground_truths(
    ground_truth_file_path: str,
) -> Dict:
    id_to_ground_truths = {}
    with jsonlines.open(ground_truth_file_path, 'r') as input_file:
        for line in input_file:
            #import pdb; pdb.set_trace()
            qid = line['question_id']
            answer = line['answers_objects'][0]['spans']
            id_to_ground_truths[qid] = answer
    return id_to_ground_truths


def load_predictions(prediction_file_path):
    with open(prediction_file_path, "r") as file:
        id_to_predictions = json.load(file)
    return id_to_predictions

# Save
def save_results(results_dict, output_path):
    output_path = output_path
    print(output_path)
    with open(output_path, "w") as file:
        json.dump(results_dict, file, indent=4)

def calculate_acc(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def evaluate_by_dicts(data_name):
    metrics = [SquadAnswerEmF1Metric()]
    id_to_predictions = load_predictions(base_pred_path + data_name+'/' + data_name+'.json')
    id_to_ground_truths = load_ground_truths('processed_data/'+data_name+'/test_subsampled.jsonl')
    total_acc = 0

    for id_ in set(id_to_ground_truths.keys()):
        ground_truth = id_to_ground_truths[id_]
        prediction = id_to_predictions[id_]

        assert isinstance(prediction, (str, list))
        if isinstance(prediction, str):
            if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
            else:
                prediction = [prediction]

        assert isinstance(prediction, (list, tuple))
        prediction = [str(e) for e in prediction]
        prediction = [answer_extractor(_prediction) for _prediction in prediction]  # Temporary.

        acc = calculate_acc(normalize_answer(prediction[0]), [normalize_answer(i) for i in ground_truth])
        total_acc = total_acc + acc
        metrics[0](prediction, ground_truth)
        
    total_acc = total_acc / len(id_to_predictions)
    evaluation_results = metrics[0].get_metric()
    evaluation_results['acc'] = total_acc        

    save_results(evaluation_results, base_pred_path + data_name+'/' +'eval_metic_result_acc.json')

def official_evaluate_by_dicts(data_name):
    id_to_predictions = load_predictions(base_pred_path + data_name+'/' + data_name+'.json')
    id_to_ground_truths = load_ground_truths('processed_data/'+data_name+'/test_subsampled.jsonl')

    question_ids = list(id_to_predictions.keys())

    for id_, prediction in id_to_predictions.items():
        if isinstance(prediction, list) and len(prediction) == 1:
            id_to_predictions[id_] = str(prediction[0])
        elif isinstance(prediction, list) and len(prediction) > 1:
            id_to_predictions[id_] = " ".join([str(e) for e in prediction])
            print("WARNING: Found a list answer prediction, concatenating it.")

    os.makedirs(".temp", exist_ok=True)

    if data_name == "hotpotqa":

        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_json(os.path.join("raw_data", "hotpotqa", "hotpot_dev_distractor_v1.json"))
        filtered_data = [datum for datum in original_data if datum["_id"] in question_ids]
        write_json(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = {
            "answer": {id_: str(prediction) for id_, prediction in id_to_predictions.items()},
            "sp": {id_: [["", 0]] for id_, _ in id_to_predictions.items()},
        }
        write_json(data, temp_prediction_file_path)

        # Run the command
        temp_ground_truth_file_path = os.path.join(os.pardir, os.pardir, temp_ground_truth_file_path)
        temp_prediction_file_path = os.path.join(os.pardir, os.pardir, temp_prediction_file_path)
        temp_output_file_path = os.path.join(os.pardir, os.pardir, ".temp", uuid.uuid4().hex)

        official_hotpotqa_evaluation_path = os.path.join("official_evaluation", "hotpotqa")
        command = (
            f"cd {official_hotpotqa_evaluation_path} ; "
            + f"python hotpot_evaluate_v1.py {temp_prediction_file_path} "
            + f"{temp_ground_truth_file_path} > {temp_output_file_path}"
        )
        status = subprocess.call(command, shell=True)
        if status != 0:
            raise Exception("Running the official evaluation script failed.")

        temp_ground_truth_file_path = temp_ground_truth_file_path.replace(
            os.path.join(os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_prediction_file_path = temp_prediction_file_path.replace(
            os.path.join(os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_output_file_path = temp_output_file_path.replace(os.path.join(os.pardir, os.pardir) + os.path.sep, "")
        if not os.path.exists(temp_output_file_path):
            raise Exception("The official evaluation output file not found.")

        with open(temp_output_file_path, "r") as file:
            metrics_ = eval(file.read().strip())
            metrics = {
                "f1": round(metrics_["f1"], 5),
                "em": round(metrics_["em"], 5),
                "precision": round(metrics_["prec"], 5),
                "recall": round(metrics_["recall"], 5),
                "count": len(id_to_predictions),
                'acc' : round(metrics_["acc"], 5),
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        save_results(metrics, base_pred_path + data_name+'/' +'eval_metic_result_acc.json')
        #return metrics

    if data_name == "2wikimultihopqa":
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_json(os.path.join("raw_data", "2wikimultihopqa", "dev.json"))
        filtered_data = [datum for datum in original_data if datum["_id"] in question_ids]
        write_json(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = {
            "answer": {id_: str(prediction) for id_, prediction in id_to_predictions.items()},
            "sp": {id_: [["", 0]] for id_, _ in id_to_predictions.items()},
            "evidence": {id_: ["", "", ""] for id_, _ in id_to_predictions.items()},
        }
        write_json(data, temp_prediction_file_path)

        # run the command
        temp_ground_truth_file_path = os.path.join(os.pardir, os.pardir, temp_ground_truth_file_path)
        temp_prediction_file_path = os.path.join(os.pardir, os.pardir, temp_prediction_file_path)
        alias_file_path = os.path.join(os.pardir, os.pardir, "raw_data", "2wikimultihopqa", "id_aliases.json")
        temp_output_file_path = os.path.join(os.pardir, os.pardir, ".temp", uuid.uuid4().hex)

        evaluation_directory = os.path.join("official_evaluation", "2wikimultihopqa")
        command = (
            f"cd {evaluation_directory} ; "
            + f"python 2wikimultihop_evaluate_v1.1.py {temp_prediction_file_path} "
            + f"{temp_ground_truth_file_path} {alias_file_path} > {temp_output_file_path}"
        )
        subprocess.call(command, shell=True)

        temp_ground_truth_file_path = temp_ground_truth_file_path.replace(
            os.path.join(os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_prediction_file_path = temp_prediction_file_path.replace(
            os.path.join(os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_output_file_path = temp_output_file_path.replace(os.path.join(os.pardir, os.pardir) + os.path.sep, "")
        if not os.path.exists(temp_output_file_path):
            raise Exception("The official evaluation output file not found.")

        with open(temp_output_file_path, "r") as file:
            metrics_ = json.loads(file.read().strip())
            metrics = {
                "f1": round(metrics_["f1"] / 100, 5),
                "em": round(metrics_["em"] / 100, 5),
                "precision": round(metrics_["prec"] / 100, 5),
                "recall": round(metrics_["recall"] / 100, 5),
                "count": len(id_to_predictions),
                'acc' : round(metrics_["acc"] / 100, 5),
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        #return metrics
        save_results(metrics, base_pred_path + data_name+'/' +'eval_metic_result_acc.json')

    if data_name == "musique":

        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_jsonl(os.path.join("raw_data", "musique", "musique_ans_v1.0_dev.jsonl"))
        original_keyed_data = {datum["id"]: datum for datum in original_data}
        filtered_data = [original_keyed_data[qid] for qid in question_ids]
        write_jsonl(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(".temp", uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = [
            {
                "id": id_,
                "predicted_answer": str(id_to_predictions[id_]),
                "predicted_support_idxs": [0, 1],
                "predicted_answerable": True,
            }
            for id_ in question_ids
        ]
        write_jsonl(data, temp_prediction_file_path)

        # run the command
        temp_ground_truth_file_path = os.path.join(os.pardir, os.pardir, temp_ground_truth_file_path)
        temp_prediction_file_path = os.path.join(os.pardir, os.pardir, temp_prediction_file_path)
        temp_output_file_path = os.path.join(os.pardir, os.pardir, ".temp", uuid.uuid4().hex)

        evaluation_directory = os.path.join("official_evaluation", "musique")
        command = (
            f"cd {evaluation_directory} ; "
            + f"python evaluate_v1.0.py {temp_prediction_file_path} {temp_ground_truth_file_path} "
            + f"--output_filepath {temp_output_file_path}"
        )
        subprocess.call(command, shell=True)

        temp_ground_truth_file_path = temp_ground_truth_file_path.replace(
            os.path.join(os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_prediction_file_path = temp_prediction_file_path.replace(
            os.path.join(os.pardir, os.pardir) + os.path.sep, ""
        )
        temp_output_file_path = temp_output_file_path.replace(os.path.join(os.pardir, os.pardir) + os.path.sep, "")

        if not os.path.exists(temp_output_file_path):
            raise Exception("The official evaluation output file not found.")

        with open(temp_output_file_path, "r") as file:
            metrics_ = json.loads(file.read().strip())
            metrics = {
                "f1": round(metrics_["answer_f1"], 3),
                "em": round(metrics_["answer_em"], 3) if "answer_em" in metrics_ else None,
                "count": len(id_to_predictions),
                "acc": round(metrics_["answer_acc"], 3),
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        #return metrics
        save_results(metrics, base_pred_path + data_name+'/' +'eval_metic_result_acc.json')
        
lst_data_name = ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']

for data_name in ['nq', 'trivia', 'squad']:
    evaluate_by_dicts(data_name)

for data_name in ['musique', 'hotpotqa', '2wikimultihopqa']:
    official_evaluate_by_dicts(data_name)