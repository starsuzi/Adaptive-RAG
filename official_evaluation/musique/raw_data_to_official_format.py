from typing import List, Dict
import argparse
import json
import os
from tqdm import tqdm

from utils.common import read_jsonl, write_jsonl, translate_id, step_placeholder


def filter_answers_in_context(answer_texts: List[str], contexts: List[Dict]) -> List[str]:

    context_text = "  ".join([
        context["wikipedia_title"] + " " + context["paragraph_text"]
        for context in contexts
    ]).lower()

    answer_texts_in_context = [
        answer_text for answer_text in answer_texts if answer_text.lower().strip() in context_text
    ]
    return answer_texts_in_context


def raw_instance_to_official_format(instance: Dict, id_to_answer_aliases: Dict) -> Dict:
    """
    Output Data Format.
    {
        "id":
        "question": "...",
        "question_decomposition": [

        ],
        "answerable": True/False,
        "answer": "...",
        "answer_aliases": ["...", "..."]
        "paragraphs": [

        ]
    }

    Prediction Format. It should be jsonl in the same order as the input.
    {
        "id": <id>,
        "predicted_support_idxs": [...],
        "predicted_answer": "...",
        "predicted_answerable": True/False
    }
    """


    def get_decomposed_question_texts(decomposed_question_text: str):

        relation_separator = ">>"
        assert relation_separator not in decomposed_question_text
        decomposed_question_text = decomposed_question_text.replace("[SEP]",  relation_separator)

        unlike_str = "<--UNLIKELY_BREAK-->"
        for index in range(5):
            if "who is #1 most listened to on spotify" not in decomposed_question_text:
                assert f"#{index+1}" not in decomposed_question_text
            decomposed_question_text = decomposed_question_text.replace(
                step_placeholder(index, is_prefix=True, strip=True),
                unlike_str
            )
            decomposed_question_text = decomposed_question_text.replace(
                step_placeholder(index, is_prefix=False, strip=True),
                f"#{index+1}"
            )

        decomposed_question_texts = [
            e.strip() for e in decomposed_question_text.split(unlike_str) if e.strip()
        ]
        return decomposed_question_texts


    paragraph2idx = {}
    updated_paragraphs = []
    for idx, context in enumerate(instance["contexts"]):
        updated_paragraphs.append({
            "idx": idx,
            "title": context["wikipedia_title"],
            "paragraph_text": context["paragraph_text"],
            "is_supporting": context["is_supporting"]
        })
        paragraph2idx[context["paragraph_text"]] = idx

    decomposed_question_texts = get_decomposed_question_texts(instance['question_text'])

    question_decomposition = []
    for decomposed_question_text, decomposed_instance in zip(
        decomposed_question_texts,
        instance["decomposed_instances"]
    ):
        sub_instance_id = decomposed_instance["id"]
        sub_answer_text = decomposed_instance["answer_text"]
        paragraph_support_idx = (
            paragraph2idx[decomposed_instance["contexts"][0]["paragraph_text"]]
            if decomposed_instance["contexts"][0]["paragraph_text"] in paragraph2idx else None
        )

        if instance["answerable"]:
            assert paragraph_support_idx is not None

        question_decomposition.append({
            "id": sub_instance_id,
            "question": decomposed_question_text,
            "answer": sub_answer_text,
             "paragraph_support_idx": paragraph_support_idx
        })


    def id2chain(chain_id: str) -> Dict:
        shape, iids = chain_id.strip().split("__")
        iids = [int(id_) for id_ in iids.split("_")]
        return {"shape": shape, "iids": iids}

    end_question_id = str(id2chain(instance['id'])['iids'][-1])
    answer_aliases = id_to_answer_aliases.get(end_question_id, [])
    answer_aliases = filter_answers_in_context(answer_aliases, instance["contexts"])
    if instance["answer_text"] in answer_aliases:
        answer_aliases.remove(instance["answer_text"])
    answer_aliases = list(set(answer_aliases))

    translated_instance = {
        "id": translate_id(instance["id"]),
        "paragraphs": updated_paragraphs,
        "question": instance['composed_question_text'],
        "question_decomposition": question_decomposition,
        "answer": instance["answer_text"],
        "answer_aliases": answer_aliases,
        "answerable": instance["answerable"],
    }

    return translated_instance

def raw_dataset_to_official_format(
        source_filepath: str,
        target_filepath: str,
        hide_labels: bool = False
    ):

    aliases_path = ".answer_aliases.json" # only applicable for musique.

    id_to_answer_aliases = {}
    with open(aliases_path, "r") as file:
        for key, value in tqdm(json.load(file).items()):
            if key in id_to_answer_aliases and sorted(value) != sorted(id_to_answer_aliases[key]):
                value = list(set(id_to_answer_aliases[key] + value))
            id_to_answer_aliases[key] = value

    source_instances = read_jsonl(source_filepath)

    translated_instances = []
    for instance in source_instances:
        translated_instance = raw_instance_to_official_format(
                instance, id_to_answer_aliases
            )
        translated_instances.append(translated_instance)

    if hide_labels:
        for translated_instance in translated_instances:
            translated_instance.pop("answer")
            translated_instance.pop("answer_aliases")
            translated_instance.pop("answerable")
            translated_instance.pop("question_composed_by", None)
            translated_instance.pop("question_decomposition")
            for paragraph in translated_instance["paragraphs"]:
                paragraph.pop("is_supporting")

    write_jsonl(translated_instances, target_filepath)


def main():

    parser = argparse.ArgumentParser(description='Convert raw dataset file to official format.')
    parser.add_argument(
        'input_filepath',
        type=str,
        help='filepath to raw dataset file.'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_filepath):
        exit(f"Filepath {args.input_filepath} not found.")

    output_filepath = "_official_format".join(os.path.splitext(args.input_filepath))
    raw_dataset_to_official_format(args.input_filepath, output_filepath)
    

if __name__ == '__main__':
    main()
