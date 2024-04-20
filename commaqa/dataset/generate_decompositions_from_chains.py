import argparse
import json
from copy import deepcopy
from math import ceil
from random import shuffle

from commaqa.inference.utils import LIST_JOINER, EOQ_MARKER, INTERQ_MARKER, ANSWER_MARKER, SIMPQ_MARKER


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Solve a ReModeL dataset using composition")
    arg_parser.add_argument("--input_json", type=str, required=True, help="Input JSON dataset files")
    arg_parser.add_argument("--chains", type=str, required=True, help="Input chains TSV file")
    arg_parser.add_argument("--decomp_json", type=str, required=False, help="Output decompositions")
    arg_parser.add_argument(
        "--max_examples",
        type=float,
        required=False,
        default=1.0,
        help="Maximum number of examples to use. " "If set to <=1.0, use as fraction.",
    )
    return arg_parser.parse_args()


def is_valid_answer(predicted_answer, gold_answer):
    if isinstance(gold_answer, list):
        gold_answer_str = LIST_JOINER.join(sorted(gold_answer))
    else:
        gold_answer_str = str(gold_answer)

    if isinstance(predicted_answer, list):
        predicted_answer_str = LIST_JOINER.join(sorted([str(s) for s in predicted_answer]))
    else:
        predicted_answer_str = str(gold_answer)
    # print(predicted_answer_str, gold_answer_str)
    return predicted_answer_str == gold_answer_str


def build_train_seqs(question_seq):
    question_seq = question_seq.strip() + " " + EOQ_MARKER
    train_seqs = [question_seq]
    while INTERQ_MARKER in question_seq:
        answer_idx = question_seq.rfind(ANSWER_MARKER)
        question_seq = question_seq[:answer_idx].strip()
        interq_idx = question_seq.rfind(INTERQ_MARKER)
        question_seq = question_seq[:interq_idx] + SIMPQ_MARKER + question_seq[interq_idx + len(INTERQ_MARKER) :]
        train_seqs.append(question_seq)
    return train_seqs


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.input_json, "r") as input_fp:
        input_json = json.load(input_fp)

    predictions_per_qid = {}
    with open(args.chains, "r") as chains_fp:
        for line in chains_fp:
            fields = line.strip().split("\t")
            qid = fields[0]
            if qid not in predictions_per_qid:
                predictions_per_qid[qid] = []
            predictions_per_qid[qid].append(fields[1:])

    decomp_json = []
    num_chains_correct_answer = 0
    num_questions_correct_chains = 0
    num_question_no_chains = 0
    num_questions = 0
    num_chains = 0
    for input_item in input_json:
        for qa_pair in input_item["qa_pairs"]:
            qid = qa_pair["id"]
            num_questions += 1
            if qid not in predictions_per_qid:
                # print(qid)
                num_question_no_chains += 1
                continue
            found_match = False
            for potential_seq in predictions_per_qid[qid]:
                num_chains += 1
                if is_valid_answer(json.loads(potential_seq[1]), qa_pair["answer"]):
                    found_match = True
                    num_chains_correct_answer += 1
                    train_seqs = build_train_seqs(potential_seq[0])
                    decomp = deepcopy(qa_pair)
                    decomp["train_seqs"] = train_seqs
                    decomp_json.append(decomp)
            if found_match:
                num_questions_correct_chains += 1

    num_questions_with_chains = num_questions - num_question_no_chains
    print("Num Questions: {}".format(num_questions))
    print(
        "Num Questions with no chains: {} ({:.2f}%)".format(
            num_question_no_chains, (num_question_no_chains * 100 / num_questions)
        )
    )
    print(
        "Num Questions with chains: {} ({:.2f}%)".format(
            num_questions_with_chains, (num_questions_with_chains * 100 / num_questions)
        )
    )
    print(
        "Num Questions with at least one correct chain: {}"
        "({:.2f}% of predicted, {:.2f}% of total)".format(
            num_questions_correct_chains,
            (num_questions_correct_chains * 100 / num_questions_with_chains),
            (num_questions_correct_chains * 100 / num_questions),
        )
    )
    print(
        "Num Chains: {}({:.2f} c per predicted, {:.2f} c per total)".format(
            num_chains, num_chains / num_questions_with_chains, num_chains / num_questions
        )
    )
    print(
        "Num Chains with correct answer: {}({:.2f}%)".format(
            num_chains_correct_answer, (num_chains_correct_answer * 100 / num_chains)
        )
    )

    if args.decomp_json:
        # sample examples here as they will be ungrouped
        if args.max_examples < 1.0:
            shuffle(decomp_json)
            decomp_json = decomp_json[: ceil(len(decomp_json) * args.max_examples)]
        elif args.max_examples > 1.0:
            shuffle(decomp_json)
            decomp_json = decomp_json[: args.max_examples]

        with open(args.decomp_json, "w") as output_fp:
            for decomp in decomp_json:
                output_fp.write(json.dumps(decomp) + "\n")
