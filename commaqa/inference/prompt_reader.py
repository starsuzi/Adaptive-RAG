import os
import json
import copy
import random
from typing import Dict, List, Union
from functools import lru_cache

random.seed(13370)  # Don't change this.


@lru_cache(maxsize=15)
def get_tokenizer(model_name):
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return AutoTokenizer.from_pretrained(model_name)


def read_prompt(
    file_path: Union[str, List[str]] = "",
    filter_by_key_values: Dict[str, List[str]] = None,
    metadata_prefix: str = "# METADATA: ",
    order_by_key: str = None,
    test_to_train_length_scale: int = 1,
    estimated_generation_length: int = 500,
    shuffle: bool = False,
    model_length_limit: int = 8000,
    tokenizer_model_name: str = "gpt2",
    removal_method: str = "last_first",  # last_first or longest_first
) -> str:

    if not file_path:
        return ""

    if isinstance(file_path, list):
        contents = []
        for _file_path in file_path:
            assert os.path.exists(_file_path), f"Filepath {_file_path} not available."
            with open(_file_path, "r") as file:
                content = file.read().strip()
                contents.append(content)
        content = "\n\n\n".join(contents)
        all_prompt_lines = [e + "\n" for e in content.strip().split("\n")]

    elif isinstance(file_path, str):
        assert os.path.exists(file_path), f"Filepath {file_path} not available."
        with open(file_path, "r") as file:
            all_prompt_lines = [e + "\n" for e in file.read().strip().split("\n")]

    else:
        raise Exception("Unexpected file_path type.")

    example = {"default": True, "lines": []}
    examples = []

    for index, line in enumerate(all_prompt_lines):

        if index == len(all_prompt_lines) - 1:
            examples.append(example)

        if line.strip().startswith(metadata_prefix):
            examples.append(example)
            metadata = line.strip().replace(metadata_prefix, "", 1)
            metadata = json.loads(metadata)
            example = copy.deepcopy(metadata)
            example["lines"] = []
        else:
            example["lines"].append(line)

    if filter_by_key_values:
        valid_examples = []
        for key, valid_values in filter_by_key_values.items():
            assert isinstance(valid_values, list)
            for example in examples:
                if not example["lines"]:
                    continue
                if key not in example:
                    print(f"WARNING: Key {key} not found in the prompt file_path ({file_path}). Skipping it.")
                    continue
                if example[key] in valid_values:
                    valid_examples.append(example)
        examples = valid_examples

    if order_by_key:
        examples = sorted(examples, key=lambda example: filter_by_key_values[key].index(example[key]))
        assert not shuffle

    prompt_examples_texts = ["".join(example["lines"]).strip() for example in examples]

    if len(prompt_examples_texts) == 1:
        # Nothing to compress. Return it as it is.
        prompt = prompt_examples_texts[0].strip()

    else:
        # Try to compress it dynamically (if needed).

        tokenizer = get_tokenizer(tokenizer_model_name)
        prompt_example_lengths = [len(tokenizer.tokenize(example_text)) for example_text in prompt_examples_texts]

        prompt_examples_original = len(prompt_examples_texts)
        prompt_examples_dropped = 0

        while prompt_example_lengths:

            estimated_test_example_length = max(prompt_example_lengths) * test_to_train_length_scale
            estimated_total_length = (
                sum(prompt_example_lengths) + estimated_test_example_length + estimated_generation_length
            )

            if estimated_total_length > model_length_limit:

                if removal_method == "longest_first":
                    max_length_index = prompt_example_lengths.index(max(prompt_example_lengths))
                    prompt_examples_texts.pop(max_length_index)
                    prompt_example_lengths.pop(max_length_index)
                    prompt_examples_dropped += 1

                elif removal_method == "last_first":
                    prompt_examples_texts.pop()
                    prompt_example_lengths.pop()
                    prompt_examples_dropped += 1

                else:
                    raise Exception(f"Unknown removal method: {removal_method}")

            else:
                break

        if not prompt_examples_texts:
            print("EXTREME WARNING: Not prompt examples remain.")

        if prompt_examples_dropped > 0:
            print(f"WARNING: Dropped {prompt_examples_dropped} / {prompt_examples_original} examples.")

        if shuffle:
            assert order_by_key is None
            random.shuffle(prompt_examples_texts)

        prompt = "\n\n\n".join([e.strip() for e in prompt_examples_texts])

    prompt = prompt.strip()
    return prompt


def fit_prompt_into_given_limit(
    original_prompt: str,
    model_length_limit: int,
    estimated_generation_length: int,
    demonstration_delimiter: str = "\n\n\n",
    shuffle: bool = False,
    remove_method: str = "first",  # first, last, random, largest
    tokenizer_model_name: str = "gpt2",
    last_is_test_example: bool = True,
):
    assert remove_method in (
        "first",
        "last",
        "random",
        "largest",
    ), "The remove_method must be from first, last, random, largest."

    demonstrations = original_prompt.strip().split(demonstration_delimiter)
    demonstrations = [demonstration.strip() for demonstration in demonstrations if demonstration.strip()]

    if len(demonstrations) <= 1:
        print("EXTREME WARNING: Found only one demonstration/example.")

    tokenizer = get_tokenizer(tokenizer_model_name)

    demonstration_sizes = [len(tokenizer.tokenize(demonstration)) for demonstration in demonstrations]

    test_example = None
    test_example_size = None
    if last_is_test_example:
        test_example = demonstrations.pop(-1)
        test_example_size = demonstration_sizes.pop(-1)

    while True:

        updated_length = sum(demonstration_sizes) + test_example_size + estimated_generation_length
        if updated_length < model_length_limit or not demonstration_sizes:
            break

        if remove_method == "first":
            remove_index = 0
        elif remove_method == "last":
            remove_index = -1
        elif remove_method == "random":
            remove_index = random.randint(0, len(demonstrations) - 1)
        elif remove_method == "largest":
            remove_index = demonstration_sizes.index(max(demonstration_sizes))
        else:
            raise Exception(f"Unexpected remove_method: {remove_method}.")

        demonstrations.pop(remove_index)
        demonstration_sizes.pop(remove_index)

        assert len(demonstrations) == len(demonstration_sizes)

    if shuffle:
        random.shuffle(demonstrations)

    if last_is_test_example:
        updated_prompt = demonstration_delimiter.join(demonstrations + [test_example])
    else:
        updated_prompt = demonstration_delimiter.join(demonstrations)

    if updated_length > model_length_limit:
        #import pdb; pdb.set_trace()
        print("EXTREME WARNING: Not enough space to even fit in even the test example.")
        updated_lines = updated_prompt.split("\n")
        while updated_lines:
            updated_lines.pop(0)
            if len(tokenizer.tokenize("\n".join(updated_lines))) <= model_length_limit:
                break
        updated_prompt = "\n".join(updated_lines)

    return updated_prompt
