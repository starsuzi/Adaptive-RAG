import os
import json
from typing import List, Dict
from pathlib import Path

import _jsonnet
from rapidfuzz import fuzz
import requests


def get_retriever_address(suffix: str = ""):
    retriever_address_config_filepath = ".retriever_address.jsonnet"
    if not os.path.exists(retriever_address_config_filepath):
        raise Exception(f"Retriver address filepath ({retriever_address_config_filepath}) not available.")
    retriever_address_config_ = json.loads(_jsonnet.evaluate_file(retriever_address_config_filepath))
    retriever_address_config = {
        "host": retriever_address_config_["host" + suffix],
        "port": retriever_address_config_["port" + suffix],
    }
    return retriever_address_config

def get_llm_server_address(llm_port_num : str):
    llm_server_address_config_filepath = ".llm_server_address.jsonnet"
    if not os.path.exists(llm_server_address_config_filepath):
        raise Exception(f"LLM Server address filepath ({llm_server_address_config_filepath}) not available.")
    llm_server_address_config = json.loads(_jsonnet.evaluate_file(llm_server_address_config_filepath))
    llm_server_address_config = {key: str(value) for key, value in llm_server_address_config.items()}
    # TODO
    #import pdb; pdb.set_trace()
    llm_server_address_config['port'] = llm_port_num
    return llm_server_address_config


def get_roscoe_server_address(suffix: str = ""):
    roscoe_server_address_config_filepath = ".roscoe_server_address.jsonnet"
    if not os.path.exists(roscoe_server_address_config_filepath):
        raise Exception(f"Retriver address filepath ({roscoe_server_address_config_filepath}) not available.")
    roscoe_server_address_config_ = json.loads(_jsonnet.evaluate_file(roscoe_server_address_config_filepath))
    roscoe_server_address_config = {
        "host": roscoe_server_address_config_["host" + suffix],
        "port": roscoe_server_address_config_["port" + suffix],
    }
    return roscoe_server_address_config


def infer_dataset_from_file_path(file_path: str) -> str:
    matching_datasets = []
    file_path = str(file_path)
    for dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        if dataset.lower() in file_path.lower():
            matching_datasets.append(dataset)
    if not matching_datasets:
        raise Exception(f"Dataset couldn't be inferred from {file_path}. No matches found.")
    if len(matching_datasets) > 1:
        raise Exception(f"Dataset couldn't be inferred from {file_path}. Multiple matches found.")
    return matching_datasets[0]


def infer_source_target_prefix(config_filepath: str, evaluation_path: str) -> str:
    source_dataset = infer_dataset_from_file_path(config_filepath)
    target_dataset = infer_dataset_from_file_path(evaluation_path)
    source_target_prefix = "_to_".join([source_dataset, target_dataset]) + "__"
    return source_target_prefix


def get_config_file_path_from_name_or_path(experiment_name_or_path: str) -> str:
    if not experiment_name_or_path.endswith(".jsonnet"):
        # It's a name
        assert (
            len(experiment_name_or_path.split(os.path.sep)) == 1
        ), "Experiment name shouldn't contain any path separators."
        matching_result = list(Path(".").rglob("**/*" + experiment_name_or_path + ".jsonnet"))
        matching_result = [
            _result
            for _result in matching_result
            if os.path.splitext(os.path.basename(_result))[0] == experiment_name_or_path
        ]
        if len(matching_result) != 1:
            exit(f"Couldn't find one matching path with the given name ({experiment_name_or_path}).")
        config_filepath = matching_result[0]
    else:
        # It's a path
        config_filepath = experiment_name_or_path
    return config_filepath


def read_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf8", errors='ignore') as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_json(instance: Dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def find_matching_paragraph_text(corpus_name: str, original_paragraph_text: str) -> str:

    retriever_address_config = get_retriever_address()
    retriever_host = str(retriever_address_config["host"])
    retriever_port = str(retriever_address_config["port"])

    params = {
        "query_text": original_paragraph_text,
        "retrieval_method": "retrieve_from_elasticsearch",
        "max_hits_count": 1,
        "corpus_name": corpus_name,
    }

    url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
    result = requests.post(url, json=params)

    if not result.ok:
        print("WARNING: Something went wrong in the retrieval. Skiping this mapping.")
        return None

    result = result.json()
    retrieval = result["retrieval"]

    for item in retrieval:
        assert item["corpus_name"] == corpus_name

    retrieved_title = retrieval[0]["title"]
    retrieved_paragraph_text = retrieval[0]["paragraph_text"]

    match_ratio = fuzz.partial_ratio(original_paragraph_text, retrieved_paragraph_text)
    #import pdb; pdb.set_trace()
    if match_ratio > 95:
        return {"title": retrieved_title, "paragraph_text": retrieved_paragraph_text}
    else:
        print(f"WARNING: Couldn't map the original paragraph text to retrieved one ({match_ratio}).")
        return None
