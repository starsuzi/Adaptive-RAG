from typing import List, Dict
import argparse
import json
import re
import os

import requests
from rapidfuzz import fuzz
from tqdm import tqdm
import _jsonnet

from lib import get_retriever_address


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    print(f"Writing {len(instances)} lines in: {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def _find_matching_paragraphs(query_title: str, query_text_substring: str, db_paragraphs: List[Dict]) -> List[Dict]:

    assert isinstance(query_title, str)
    assert isinstance(query_text_substring, str)

    matching_paragraphs = []
    for paragraph in db_paragraphs:

        title_exact_match = query_title.lower().strip() == paragraph["title"].lower().strip()
        paragraph_text_match_score = fuzz.partial_ratio(query_text_substring, paragraph["paragraph_text"])

        if title_exact_match and paragraph_text_match_score > 95:
            matching_paragraphs.append(paragraph)

    return matching_paragraphs


class Retriever:
    def __init__(self, host: str, port: int, source_corpus_name: str) -> None:
        self._host = host
        self._port = port
        assert source_corpus_name in ("hotpotqa", "2wikimultihopqa", "musique", "iirc")
        self._source_corpus_name = source_corpus_name

    def retrieve(self, allowed_title: str, query_text: str) -> List[Dict]:

        params = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": query_text,
            "max_hits_count": 50,
            "document_type": "paragraph_text",
            "allowed_titles": [allowed_title],
            "corpus_name": self._source_corpus_name,
        }
        url = self._host.rstrip("/") + ":" + str(self._port) + "/retrieve"
        result = requests.post(url, json=params)

        selected_titles = []
        selected_paras = []
        unique_retrieval = []
        if result.ok:

            result = result.json()
            retrieval = result["retrieval"]

            for retrieval_item in retrieval:

                if retrieval_item["corpus_name"] != self._source_corpus_name:
                    raise Exception(
                        f"The retrieved corpus name {retrieval_item['corpus_name']} "
                        f"doesn't match {self._source_corpus_name}."
                    )

                if (  # This was changed post-hoc to include both conditions.
                    retrieval_item["title"] in selected_titles and retrieval_item["paragraph_text"] in selected_paras
                ):
                    continue

                selected_titles.append(retrieval_item["title"])
                selected_paras.append(retrieval_item["paragraph_text"])
                unique_retrieval.append(retrieval_item)
        else:
            raise Exception("Retrieval request did not succeed.")

        return unique_retrieval


def attach_data_annotations(
    processed_data: List[Dict],
    annotations: List[Dict],
    retriever: Retriever,
) -> List[Dict]:

    id_to_annotation = {annotation["question_id"]: annotation for annotation in annotations}
    assert len(id_to_annotation) == len(annotations), "Looks like there are duplicate qid annotations."

    annotated_processed_data = []
    for instance in tqdm(processed_data):
        annotation = id_to_annotation.pop(instance["question_id"], None)

        if not annotation:
            continue

        assert instance["question_id"] == annotation["question_id"]
        question_id = instance["question_id"]

        question_match_score = fuzz.ratio(instance["question_text"], annotation["question_text"])
        if question_match_score < 95:
            print(
                "WARNING the following questions may not be same. Check manually : "
                f'{instance["question_text"]} >>> {annotation["question_text"]}'
            )

        instance["question_text"] = annotation["question_text"]
        instance["reasoning_steps"] = annotation["reasoning_steps"]
        reasoning_steps = instance["reasoning_steps"]

        answer_regex = r".*answer is: (.*)\."
        assert re.match(answer_regex, reasoning_steps[-1]["cot_sent"])
        extracted_answer = re.sub(answer_regex, r"\1", reasoning_steps[-1]["cot_sent"])

        gold_answer = instance["answers_objects"][0]["spans"][0]
        if extracted_answer != gold_answer:
            print(
                f"WARNING: The extracted answer doesn't perfectly match the gold answer. "
                f"{extracted_answer} != {gold_answer}"
            )

        # Doing this as we want CoT answers to be the same as Direct answer.
        gold_answer = extracted_answer
        instance["answers_objects"][0]["spans"][0] = gold_answer

        if retriever._source_corpus_name == "iirc":
            # In IIRC, the "context" key contains gold snippets and not paragraphs. So retrieve them instead.
            # The "pinned_contexts" key contains fixed RC on which the question is anchored.
            context_paragraphs = instance.get("pinned_contexts", [])
        else:
            context_paragraphs = instance["contexts"]

        for paragraph in instance["contexts"]:
            assert not paragraph["paragraph_text"].startswith("Title: ")
            assert not paragraph["paragraph_text"].startswith("Wikipedia Title: ")

        text_populated_reasoning_steps = []
        for reasoning_step in reasoning_steps:

            # First, try to match it to the context_paragraphs.
            assert len(reasoning_step["paragraphs"]) == 1  # TODO: Make it single entry only.
            gold_paragraph = reasoning_step["paragraphs"][0]

            assert "title" in gold_paragraph, f"Field `title` missing in annotation for {question_id}"
            assert "text_substring" in gold_paragraph, f"Field `text_substring` missing in annotation for {question_id}"

            if not gold_paragraph["title"] or not gold_paragraph["text_substring"]:
                assert not gold_paragraph["title"] and not gold_paragraph["text_substring"]
                gold_paragraph["paragraph_text"] = None
                text_populated_reasoning_steps.append(reasoning_step)
                continue

            matching_paragraphs = _find_matching_paragraphs(
                gold_paragraph["title"], gold_paragraph["text_substring"], context_paragraphs
            )

            assert len(matching_paragraphs) < 2

            # Otherwise try to do a match based on retrieved paragraphs.
            if not matching_paragraphs:
                retrieved_paragraphs = retriever.retrieve(gold_paragraph["title"], gold_paragraph["text_substring"])
                matching_paragraphs = _find_matching_paragraphs(
                    gold_paragraph["title"], gold_paragraph["text_substring"], retrieved_paragraphs
                )

            if not matching_paragraphs:
                print("WARNING: Couldn't find any match for the annotated paragraph.")
                continue

            assert len(matching_paragraphs) == 1
            matching_paragraph = matching_paragraphs[0]

            assert gold_paragraph["title"].lower() == matching_paragraph["title"].lower()
            gold_paragraph["paragraph_text"] = matching_paragraph["paragraph_text"]

            text_populated_reasoning_steps.append(reasoning_step)

        assert len(text_populated_reasoning_steps) == len(reasoning_steps)
        instance["reasoning_steps"] = text_populated_reasoning_steps
        annotated_processed_data.append(instance)

    return annotated_processed_data


def main():

    parser = argparse.ArgumentParser(description="Attach annotations to the processed data.")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", choices={"hotpotqa", "2wikimultihopqa", "musique", "iirc"}
    )
    args = parser.parse_args()

    annotations_file_path = os.path.join("prompt_generator", "data_annotations", args.dataset_name + ".jsonnet")
    processed_data_file_path = os.path.join("processed_data", args.dataset_name, "train.jsonl")
    output_file_path = os.path.join("processed_data", args.dataset_name, "annotated_only_train.jsonl")

    annotations = json.loads(_jsonnet.evaluate_file(annotations_file_path))
    processed_data = read_jsonl(processed_data_file_path)

    retriever_address_config = get_retriever_address()

    retriever = Retriever(
        host=retriever_address_config["host"],
        port=retriever_address_config["port"],
        source_corpus_name=args.dataset_name,
    )

    attached_data_annotations = attach_data_annotations(processed_data, annotations, retriever)
    write_jsonl(attached_data_annotations, output_file_path)


if __name__ == "__main__":
    main()
