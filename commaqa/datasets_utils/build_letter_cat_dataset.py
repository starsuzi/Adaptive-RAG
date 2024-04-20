"""
Script to build dataset for concatenating letters from words.
E.g.
Take the letters at position 3 of the words in "Nestor Geng Duran" and concatenate them.

Sample usage:
python commaqa/datasets_utils/build_letter_cat_dataset.py \
    --input_first_names configs/letter_datasets/first_names.txt \
    --input_last_names configs/letter_datasets/last_names.txt \
    --output datasets/letter_cat/n3_e20_pos3.txt \
    --num_words 3 --num_examples 20 --position 3 --add_space
"""

import argparse
import json
import math
import random


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Create dataset for letter concatenation")
    arg_parser.add_argument("--input_first_names", type=str, required=True, help="Input file of first names")
    arg_parser.add_argument("--input_last_names", type=str, required=True, help="Input file of last names")
    arg_parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    arg_parser.add_argument("--num_words", type=int, default=3, required=False, help="Number of words")
    arg_parser.add_argument(
        "--num_examples", type=int, default=20, required=False, help="Number of examples in dataset"
    )
    arg_parser.add_argument(
        "--position",
        type=int,
        default=1,
        required=False,
        help="Position of letter to concatenate(-1 used for last letter, "
        "-100 for random sampling between 1 and 7 and -1.",
    )
    arg_parser.add_argument(
        "--add_space",
        default=False,
        required=False,
        action="store_true",
        help="Set to add spaces in the output string. 'n i e' vs 'nie'",
    )
    return arg_parser.parse_args()


def words_with_min_length(words, length):
    output = []
    for w in words:
        if len(w) > length >= 0:
            output.append(w)
        elif length < 0 and len(w) >= abs(length):
            # e.g. if length is -1, word should have at least one letter
            output.append(w)
    return output


def create_word_concat_question(pos, name):
    if pos == -1:
        pos_str = "the last letters"
    elif pos == -2:
        pos_str = "the second last letters"
    else:
        pos_str = "the letters at position {}".format(pos + 1)
    return 'Take {} of the words in "{}" and concatenate them.'.format(pos_str, name)


def create_single_word_question(pos, name):
    if pos == -1:
        pos_str = "the last letter"
    elif pos == -2:
        pos_str = "the second last letter"
    else:
        pos_str = "the letter at position {}".format(pos + 1)
    return 'Return {} of the word "{}".'.format(pos_str, name)


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.input_first_names, "r") as input_fp:
        first_names = [f.strip() for f in input_fp.readlines() if " " not in f.strip() and "-" not in f]

    with open(args.input_last_names, "r") as input_fp:
        last_names = [f.strip() for f in input_fp.readlines() if " " not in f.strip() and "-" not in f]

    qa_pairs = []
    for eg_idx in range(args.num_examples):
        # construct name
        if args.position == -100:
            pos = random.choice(list(range(7)) + [-1])
        else:
            pos = args.position

        valid_first_names = words_with_min_length(first_names, pos)
        valid_last_names = words_with_min_length(last_names, pos)
        if len(valid_first_names) == 0 or len(valid_last_names) == 0:
            raise ValueError("No names with length exceeding {}. Choose a different value for" " position".format(pos))
        words = []
        for n_idx in range(args.num_words):
            if n_idx < math.floor(args.num_words / 2):
                words.append(random.choice(valid_first_names))
            else:
                words.append(random.choice(valid_last_names))

        name = " ".join(words)
        delim = " " if args.add_space else ""
        answer = delim.join([w[pos] for w in words])
        drop_answer = {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": [answer]}
        if args.num_words == 1:
            question = create_single_word_question(pos, name)
        else:
            question = create_word_concat_question(pos, name)
        qa_pairs.append(
            {"question": question, "answer": drop_answer, "query_id": str(eg_idx), "name": name, "words": words}
        )

    output_json = {"1": {"passage": "", "qa_pairs": qa_pairs}}
    with open(args.output, "w") as output_fp:
        output_fp.write(json.dumps(output_json, indent=2))
