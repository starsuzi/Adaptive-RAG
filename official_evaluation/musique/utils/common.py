from typing import Dict, List
import re
import json
from tqdm import tqdm
import string
import os

from utils.constants import (
    CONSTITUENT_QUESTION_START, CONSTITUENT_QUESTION_END,
    REPLACEMENT_QUESTION_START, REPLACEMENT_QUESTION_END,
)


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

    return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()


def step_placeholder(index: int, is_prefix: bool, strip: bool = False) -> str:

    if is_prefix:
        output = f"{CONSTITUENT_QUESTION_START} {index} {CONSTITUENT_QUESTION_END} "
    else:
        output = f"{REPLACEMENT_QUESTION_START} {index} {REPLACEMENT_QUESTION_END} "

    if strip:
        output = output.strip()
    return output


def write_jsonl(instances: List[Dict], file_path: str) -> None:
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    print(f"Writing in {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in tqdm(file) if line.strip()]
    return instances

def translate_id(key: str) -> str:
    # Naming convention change for question-ids in the released dataset format.
    namechange = {
        "double": "2hop", "triple_ii": "3hop1",
        "triple_io": "3hop2", "quadruple_iii": "4hop1",
        "quadruple_iot": "4hop2", "quadruple_ioh1": "4hop3", "quadruple_ioh2": "4hop4"
    }
    assert sum([original in key for original, new in namechange.items()]) == 1
    for original, new in namechange.items():
        key = key.replace(original, new)
    return key
