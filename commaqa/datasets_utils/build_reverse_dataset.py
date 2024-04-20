# Reverse the sequence "card, stamp, book, water, glasses".

import json
import math
import random
import string

# TODO make configurable
reverse_letters = True
num_examples = 100
length_range = range(4, 5)


def main():
    with open("configs/reverse_datasets/wordlist.txt", "r") as f:
        wordlist = list(map(str.strip, f))
    for list_length in length_range:
        if reverse_letters:
            input_arr = string.ascii_lowercase
        else:
            input_arr = wordlist
        # number of permutations
        permutation_count = math.perm(len(input_arr), list_length)
        # select num_examples permutation indexes
        permutation_idxs = random.sample(range(permutation_count), num_examples)
        # create num_examples permutations
        sublists = (get_permutation(i, input_arr, list_length) for i in permutation_idxs)
        qa_pairs = list()
        for sublist in sublists:
            if reverse_letters:
                question_word = "".join(sublist)
                question = "Reverse the letters in the word {}".format(question_word)
                answer = "".join(reversed(sublist))
            else:
                comma_separated_sublist = ", ".join(sublist)
                question = f'Reverse the sequence "{comma_separated_sublist}".'
                answer = ", ".join(reversed(sublist))
            qa_pairs.append((question, answer))
        drop = {
            "reverseqa": {
                "passage": "",
                "qa_pairs": [
                    {
                        "question": question,
                        "answer": {
                            "number": "",
                            "date": {"day": "", "month": "", "year": ""},
                            "spans": [answer],
                        },
                        "query_id": str(i),
                        "validated_answers": [],
                    }
                    for i, (question, answer) in enumerate(qa_pairs)
                ],
            }
        }
        filename = "reverse_{}_{}_{}.json".format(list_length, num_examples, "L" if reverse_letters else "W")
        with open(filename, "w") as f:
            json.dump(drop, f, indent=2)


def get_permutation(i, lst, length):
    permutation = list()
    for _ in range(length):
        i, idx = divmod(i, len(lst))
        permutation.append(lst[idx])
        # remove to prevent duplicates
        lst = lst[:idx] + lst[idx + 1 :]
    return permutation


if __name__ == "__main__":
    main()
