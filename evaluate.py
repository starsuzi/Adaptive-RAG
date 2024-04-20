import re
import os
import json
import uuid
import subprocess
import argparse
from typing import Dict, Any
import string

import _jsonnet
from commaqa.inference.constants import PREDICTION_TYPES, READER_NAME_CLASS
from lib import (
    get_retriever_address,
    get_llm_server_address,
    infer_source_target_prefix,
    infer_dataset_from_file_path,
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


def evaluate_by_dicts(
    prediction_type: str,
    id_to_ground_truths: Dict[str, Any],
    id_to_predictions: Dict[str, Any],
    dataset: str,
) -> Dict:
    if prediction_type == "answer":
        if dataset in ['hotpotqa', '2wikimultihopqa', 'musique', 'iirc']:
            metrics = [DropAnswerEmAndF1(), SupportEmF1Metric(do_normalize_answer=True)]
        else:
            metrics = [SquadAnswerEmF1Metric(), SupportEmF1Metric(do_normalize_answer=True)]
    elif prediction_type in ("titles", "pids", "real_pids"):
        metrics = [SupportEmF1Metric()]
    elif prediction_type in ("paras"):
        metrics = [AnswerSupportRecallMetric()]

    for id_ in set(id_to_ground_truths.keys()):
        ground_truth = id_to_ground_truths[id_]
        prediction = id_to_predictions[id_]

        assert isinstance(prediction, (str, list))
        if prediction_type == "answer" and isinstance(prediction, str):
            if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
            else:
                prediction = [prediction]

        assert isinstance(prediction, (list, tuple))
        prediction = [str(e) for e in prediction]

        if prediction_type == "answer":
            prediction = [answer_extractor(_prediction) for _prediction in prediction]  # Temporary.
            metrics[0](prediction, [ground_truth])
            metrics[1](prediction, ground_truth)
        elif prediction_type in ("titles", "pids", "real_pids"):
            metrics[0](prediction, ground_truth)
        elif prediction_type in ("paras"):
            predicted_paras = [
                " ".join([eval(prediction_)["title"], eval(prediction_)["paragraph_text"]])
                for prediction_ in prediction
            ]
            metrics[0](predicted_paras, ground_truth)

    evaluation_results = metrics[0].get_metric()

    if prediction_type == "answer":
        evaluation_results_ = metrics[1].get_metric()
        evaluation_results["sp_em"] = evaluation_results_["title_em"]
        evaluation_results["sp_f1"] = evaluation_results_["title_f1"]
        evaluation_results["sp_precision"] = evaluation_results_["title_precision"]
        evaluation_results["sp_recall"] = evaluation_results_["title_recall"]

    return evaluation_results


def official_evaluate_by_dicts(
    prediction_type: str, id_to_predictions: Dict[str, Any], id_to_ground_truths: Dict[str, Any], dataset: str
) -> Dict:

    if prediction_type != "answer":
        # official evaluation is not available for non answer prediction.
        return evaluate_by_dicts(prediction_type, id_to_ground_truths, id_to_predictions, dataset)

    question_ids = list(id_to_predictions.keys())

    for id_, prediction in id_to_predictions.items():
        if isinstance(prediction, list) and len(prediction) == 1:
            id_to_predictions[id_] = str(prediction[0])
        elif isinstance(prediction, list) and len(prediction) > 1:
            id_to_predictions[id_] = " ".join([str(e) for e in prediction])
            print("WARNING: Found a list answer prediction, concatenating it.")

    os.makedirs(".temp", exist_ok=True)

    if dataset == "hotpotqa":

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
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        return metrics

    if dataset == "2wikimultihopqa":

        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(".temp", uuid.uuid4().hex)
        original_data = read_json(os.path.join("raw_data", "2wikimultihopqa", "dev.jsonl"))
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
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        return metrics

    if dataset == "musique":

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
            }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)
        os.remove(temp_output_file_path)

        return metrics

    if dataset == "iirc":
        return evaluate_by_dicts("answer", id_to_ground_truths, id_to_predictions, dataset)


def load_experiment_config(config_file_path: str, args):
    env_variables = {}
    retriever_address = get_retriever_address()
    env_variables["RETRIEVER_HOST"] = str(retriever_address["host"])
    env_variables["RETRIEVER_PORT"] = str(retriever_address["port"])
    llm_server_address = get_llm_server_address(args.llm_port_num)
    env_variables["LLM_SERVER_HOST"] = str(llm_server_address["host"])
    env_variables["LLM_SERVER_PORT"] = str(llm_server_address["port"])
    config = json.loads(_jsonnet.evaluate_file(config_file_path, ext_vars=env_variables))
    return config


def load_ground_truths(
    experiment_config: Dict,
    ground_truth_file_path: str,
    question_type_key: str = None,
    question_type_value: str = None,
) -> Dict:

    # Load the config
    reader_config = experiment_config["reader"]
    reader_name = reader_config.pop("name")
    reader = READER_NAME_CLASS[reader_name](**reader_config)

    # Prep prediction_type and reader
    prediction_type = experiment_config["prediction_type"]
    if prediction_type in ("titles", "pids", "real_pids") and reader_name != "multi_para_rc":
        exit("The titles and pids prediction evaluation is only supported for multi_para_rc reader.")

    if prediction_type in ("titles", "pids", "real_pids", "paras"):
        reader.add_paras = False
        reader.add_gold_paras = True
        reader.add_pinned_paras = False
        reader.remove_pinned_para_titles = True
        reader.add_paras_from_files = None

    # prep ground_truths
    id_to_ground_truths = {}
    for example in reader.read_examples(ground_truth_file_path):

        if question_type_key is not None or question_type_value is not None:
            if question_type_key is None or question_type_value is None:
                raise Exception("Both question type key and value must be passed if any one of them is passed.")
            if question_type_key not in example["metadata"]:
                raise Exception(f"Key {question_type_key} not present in the example instance.")

            if example["metadata"][question_type_key] != question_type_value:
                continue

        id_ = example["qid"]
        if prediction_type in ("answer", "paras"):
            id_to_ground_truths[id_] = example["answer"]
        elif prediction_type == "titles":
            id_to_ground_truths[id_] = example["titles"]
        elif prediction_type == "pids":
            id_to_ground_truths[id_] = example["pids"]
        elif prediction_type == "real_pids":
            id_to_ground_truths[id_] = example["real_pids"]
        else:
            raise Exception("Unknown prediction_type.")

    return id_to_ground_truths
    

def load_predictions(prediction_file_path: str) -> Dict:
    with open(prediction_file_path, "r") as file:
        id_to_predictions = json.load(file)
    return id_to_predictions


def parse_multi_step_retrieval_predictions(file_path: str) -> Dict:
    # NOTE: This should only be run on multi_step retrieval predictions

    if "multi_step" not in file_path or "retrieval" not in file_path:
        print("WARNING: This may not be multi_step retrieval prediction based on the file_path.")

    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    parsed_dicts = []

    is_qid = True
    is_question_text = False
    question_text = None
    question_id = None
    generated_titles = []
    projected_titles = []
    final_titles = []
    cot_sents = []
    encountered_pids = False  # To ensure this script is run only on retrieval models

    global_is_llm_retrieval = None

    for index, line in enumerate(lines):

        if is_question_text:
            question_text = line.strip()

        if is_qid:
            question_id = line.strip()
            is_question_text = True
        else:
            is_question_text = False

        if line.startswith("S: "):
            assert encountered_pids, "Looks like this is not a retrieval prediction."
            parsed_dicts.append(
                {
                    "qid": question_id,
                    "question_text": question_text,
                    "generated_titles": generated_titles,
                    "projected_titles": projected_titles,
                    "final_titles": final_titles,
                    "cot_sents": cot_sents,
                }
            )
            is_qid = True
            is_question_text = False
            question_text = None
            question_id = None
            generated_titles = []
            projected_titles = []
            final_titles = []
            cot_sents = []
            encountered_pids = False
        else:
            is_qid = False

        if line.startswith('A: ["pid'):
            encountered_pids = True

        if line.startswith("A: Exit? No.") or line.startswith('A: ["pid'):

            is_llm_retrieval = (
                (lines[index - 2].startswith("A: ") and not lines[index - 2].startswith("A: Exit? No."))
                and (lines[index - 3].startswith("A: ") and not lines[index - 3].startswith("A: Exit? No."))
                and (lines[index - 4].startswith("A: ") and not lines[index - 4].startswith("A: Exit? No."))
            )

            if global_is_llm_retrieval is None:
                global_is_llm_retrieval = is_llm_retrieval
            else:
                assert global_is_llm_retrieval == is_llm_retrieval

            if is_llm_retrieval:
                # the generation and projection isn't cummulated.
                generated_titles.append(re.findall(r'"(.+?)"[,\]]', lines[index - 4].replace("A: ", "", 1)))
                projected_titles.append(re.findall(r'"(.+?)"[,\]]', lines[index - 3].replace("A: ", "", 1)))

            final_titles_so_far = json.loads(lines[index - 2].replace("A: ", "", 1))
            cot_sents_so_far = lines[index - 1].replace("A: ", "", 1)
            final_titles.append(final_titles_so_far)
            cot_sents.append(cot_sents_so_far)

    id_to_faired_parsed_dict = {}
    for parsed_dict in parsed_dicts:

        assert len(parsed_dict["final_titles"]) == len(parsed_dict["cot_sents"])
        if not parsed_dict["generated_titles"]:
            parsed_dict["generated_titles"] = [None] * len(parsed_dict["final_titles"])
        if not parsed_dict["projected_titles"]:
            parsed_dict["projected_titles"] = [None] * len(parsed_dict["final_titles"])

        steps = []
        last_so_far_final_titles = []
        last_so_far_cot_sent = ""
        for generated_titles, projected_titles, so_far_final_titles, so_far_cot_sents in zip(
            parsed_dict["generated_titles"],
            parsed_dict["projected_titles"],
            parsed_dict["final_titles"],
            parsed_dict["cot_sents"],
        ):
            new_titles = [title for title in so_far_final_titles if title not in last_so_far_final_titles]
            if last_so_far_cot_sent:
                assert so_far_cot_sents.count(last_so_far_cot_sent) == 1
            new_cot_sent = so_far_cot_sents.replace(last_so_far_cot_sent, "").strip()
            step = {
                "generated_titles": generated_titles,
                "projected_titles": projected_titles,
                "new_final_titles": new_titles,
                "new_cot_sent": new_cot_sent,
            }
            steps.append(step)
            last_so_far_final_titles = so_far_final_titles
            last_so_far_cot_sent = so_far_cot_sents

        id_to_faired_parsed_dict[parsed_dict["qid"]] = {"question_text": parsed_dict["question_text"], "steps": steps}

    return id_to_faired_parsed_dict


def main():
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument("experiment_name_or_path", type=str, help="experiment_name_or_path")
    parser.add_argument("evaluation_path", type=str, help="evaluation_path")
    parser.add_argument("--prediction-type", type=str, help="optional prediction-type", choices=PREDICTION_TYPES)
    parser.add_argument(
        "--prediction-suffix", type=str, help="optional suffix for the prediction directory.", default=""
    )
    parser.add_argument(
        "--question-type-key-value", type=str, help="':' separated question-type-key-value.", default=None
    )
    parser.add_argument("--only-print", action="store_true", default=False, help="only print don't run evaluation")
    parser.add_argument(
        "--official", action="store_true", default=False, help="use official eval scripts when available."
    )
    parser.add_argument('--set_name', type=str, help="set_name", required=True)
    # TODO
    # llm_port_num
    parser.add_argument(
        '--llm_port_num', type=str, help="llm_port_num", required=True
    )
    args = parser.parse_args()

    config_filepath = get_config_file_path_from_name_or_path(args.experiment_name_or_path)
    experiment_name = os.path.splitext(os.path.basename(config_filepath))[0]

    prediction_directory = os.path.join("predictions", args.set_name, experiment_name + args.prediction_suffix)
    prediction_file_name = os.path.splitext(os.path.basename(args.evaluation_path))[0]
    prediction_file_name = infer_source_target_prefix(config_filepath, args.evaluation_path) + prediction_file_name
    prediction_file_path = os.path.join(prediction_directory, "prediction__" + prediction_file_name + ".json")

    if not os.path.exists(prediction_file_path):
        exit(f"The prediction_file_path {prediction_file_path} is not available.")

    official_prefix = "official_" if args.official else ""
    save_metrics_path = os.path.join(
        prediction_directory, official_prefix + "evaluation_metrics__" + prediction_file_name + ".json"
    )
    if args.only_print:
        if not os.path.exists(save_metrics_path):
            exit("Asked to print the metrics, but the metrics file_path is not available.")
        with open(save_metrics_path, "r") as file:
            print(file.read())
        exit()

    # get prediction_type
    experiment_config = load_experiment_config(config_filepath, args)
    prediction_type = experiment_config["prediction_type"]

    # prep ground_truths
    question_type_key = question_type_value = None
    if args.question_type_key_value is not None:
        if args.question_type_key_value.count(":") != 1:
            raise Exception("The question_type_key_value must be : separated.")
        question_type_key, question_type_value = args.question_type_key_value.split(":")
        question_type_key = question_type_key.strip()
        question_type_value = question_type_value.strip()

    id_to_ground_truths = load_ground_truths(
        experiment_config,
        args.evaluation_path,
        question_type_key=question_type_key,
        question_type_value=question_type_value,
    )

    # prep predictions
    id_to_predictions = load_predictions(prediction_file_path)

    if question_type_value is not None:
        id_to_predictions = {
            id_: prediction for id_, prediction in id_to_predictions.items() if id_ in id_to_ground_truths.keys()
        }

    # verify equality
    if set(id_to_ground_truths.keys()) != set(id_to_predictions.keys()):
        exit("Ids in input examples and predictions don't match.")

    # evaluate
    if args.official:
        dataset = infer_dataset_from_file_path(args.evaluation_path)
        evaluation_results = official_evaluate_by_dicts(
            prediction_type=prediction_type,
            id_to_predictions=id_to_predictions,
            id_to_ground_truths=id_to_ground_truths,
            dataset=dataset,
        )
    else:
        dataset = infer_dataset_from_file_path(args.evaluation_path)
        evaluation_results = evaluate_by_dicts(
            prediction_type=prediction_type,
            id_to_predictions=id_to_predictions,
            id_to_ground_truths=id_to_ground_truths,
            dataset=dataset,
        )
    print(json.dumps(evaluation_results, indent=4))

    # To be able to reproduce the same result, save git-hash
    git_hash_filepath = os.path.join(prediction_directory, "git_hash__" + prediction_file_name + ".txt")
    if os.path.exists(git_hash_filepath):
        with open(git_hash_filepath, "r") as file:
            git_hash = file.read().strip()
        evaluation_results["git_hash"] = git_hash

    # Save the evaluation metrics
    print(f"Saving metrics in {save_metrics_path}")
    with open(save_metrics_path, "w") as file:
        json.dump(evaluation_results, file, indent=4)
    print(evaluation_results)

    # Save the ground_truth used in the same json/dict format (just for convenience)
    ground_truth_in_dict_file_path = os.path.join(
        prediction_directory, "ground_truth__" + prediction_file_name + ".json"
    )
    with open(ground_truth_in_dict_file_path, "w") as file:
        json.dump(id_to_ground_truths, file, indent=4)

    # TODO
    # Save the zero single multi classification results
    id_to_zero_single_multi_classification = {}
    dict_zero_single_multi = {
        'ircot' : 'multi',
        'oner' : 'single',
        'nor' : 'zero'
    }
    lst_zero_single_multi = []
    zero_single_multi = dict_zero_single_multi[experiment_name.split('_')[0]]
    lst_zero_single_multi.append(zero_single_multi)
    if 'bm25_retrieval_count' in experiment_name :
        retrieval_count = [i for i in experiment_name.split('___') if 'bm25_retrieval_count' in i][0].split('__')[1]
        lst_zero_single_multi.append(retrieval_count)
    for qid in id_to_ground_truths.keys():
        lst_gold_ans = id_to_ground_truths[qid]
        pred = id_to_predictions[qid]
        
        for gold_ans in lst_gold_ans:
            if normalize_answer(gold_ans) == normalize_answer(pred):
                id_to_zero_single_multi_classification[qid] = lst_zero_single_multi

    zero_single_multi_classification_path = os.path.join(
        prediction_directory, "zero_single_multi_classification__" + prediction_file_name + ".json"
    )    
    with open(zero_single_multi_classification_path, "w") as file:
        json.dump(id_to_zero_single_multi_classification, file, indent=4)

if __name__ == "__main__":
    main()
