import os
import json
from collections import Counter
from typing import List, Dict

from tqdm import tqdm
from datasets import load_dataset


def write_hotpotqa_instances_to_filepath(instances: List[Dict], full_filepath: str):

    max_num_tokens = 1000  # clip later.

    hop_sizes = Counter()
    print(f"Writing in: {full_filepath}")
    with open(full_filepath, "w") as full_file:
        for raw_instance in tqdm(instances):

            # Generic RC Format
            processed_instance = {}
            processed_instance["dataset"] = "hotpotqa"
            processed_instance["question_id"] = raw_instance["id"]
            processed_instance["question_text"] = raw_instance["question"]
            processed_instance["level"] = raw_instance["level"]
            processed_instance["type"] = raw_instance["type"]

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            processed_instance["answers_objects"] = [answers_object]

            raw_context = raw_instance.pop("context")
            supporting_titles = raw_instance.pop("supporting_facts")["title"]

            title_to_paragraph = {
                title: "".join(text) for title, text in zip(raw_context["title"], raw_context["sentences"])
            }
            paragraph_to_title = {
                "".join(text): title for title, text in zip(raw_context["title"], raw_context["sentences"])
            }

            gold_paragraph_texts = [title_to_paragraph[title] for title in supporting_titles]
            gold_paragraph_texts = set(list(gold_paragraph_texts))

            paragraph_texts = ["".join(paragraph) for paragraph in raw_context["sentences"]]
            paragraph_texts = list(set(paragraph_texts))

            processed_instance["contexts"] = [
                {
                    "idx": index,
                    "title": paragraph_to_title[paragraph_text].strip(),
                    "paragraph_text": paragraph_text.strip(),
                    "is_supporting": paragraph_text in gold_paragraph_texts,
                }
                for index, paragraph_text in enumerate(paragraph_texts)
            ]

            supporting_contexts = [context for context in processed_instance["contexts"] if context["is_supporting"]]
            hop_sizes[len(supporting_contexts)] += 1

            for context in processed_instance["contexts"]:
                context["paragraph_text"] = " ".join(context["paragraph_text"].split(" ")[:max_num_tokens])

            full_file.write(json.dumps(processed_instance) + "\n")

    print(f"Hop-sizes: {str(hop_sizes)}")


if __name__ == "__main__":

    dataset = load_dataset("hotpot_qa", "distractor")

    directory = os.path.join("processed_data", "hotpotqa")
    os.makedirs(directory, exist_ok=True)

    processed_full_filepath = os.path.join(directory, "train.jsonl")
    write_hotpotqa_instances_to_filepath(dataset["train"], processed_full_filepath)

    processed_full_filepath = os.path.join(directory, "dev.jsonl")
    write_hotpotqa_instances_to_filepath(dataset["validation"], processed_full_filepath)
