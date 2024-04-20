import argparse
import json
from copy import deepcopy
from math import ceil
from random import shuffle

from commaqa.configs.predicate_language_config import ModelQuestionConfig
from commaqa.dataset.utils import nonempty_answer
from commaqa.execution.operation_executer import OperationExecuter
from commaqa.execution.utils import build_models


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Solve a ReModeL dataset using composition")
    arg_parser.add_argument("--input_json", type=str, required=True, help="Input JSON dataset files")
    arg_parser.add_argument("--pred_json", type=str, required=False, help="Output predictions")
    arg_parser.add_argument("--decomp_json", type=str, required=False, help="Output decompositions")
    arg_parser.add_argument(
        "--max_examples",
        type=float,
        required=False,
        default=1.0,
        help="Maximum number of examples to use. " "If set to <=1.0, use as fraction.",
    )
    return arg_parser.parse_args()


def build_chain(prev_chain, operation, model, question):
    return prev_chain + " QS: ({}) [{}] {}".format(operation, model, question)


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.input_json, "r") as input_fp:
        input_json = json.load(input_fp)

    pred_json = {}
    decomp_json = []
    for input_item in input_json:
        kb = input_item["kb"]
        model_configurations = {}
        for model_name, configs in input_item["pred_lang_config"].items():
            model_configurations[model_name] = [ModelQuestionConfig(config) for config in configs]
        model_lib = build_models(model_configurations, kb)

        executor = OperationExecuter(model_lib)
        for qa_pair in input_item["qa_pairs"]:
            qid = qa_pair["id"]
            # use oracle decomposition
            curr_assignment = {}
            last_answer = ""
            train_seqs = []
            prev_chain = " QC: " + qa_pair["question"]
            for idx, step in enumerate(qa_pair["decomposition"]):
                train_seq = build_chain(
                    prev_chain=prev_chain, operation=step["op"], model=step["m"], question=step["q"]
                )
                train_seqs.append(train_seq)
                answers, facts_used = executor.execute_operation(
                    operation=step["op"], model=step["m"], question=step["q"], assignments=curr_assignment
                )
                last_answer = answers
                if not nonempty_answer(answers):
                    print("no answer!")
                    print(step, curr_assignment, kb)
                    break
                prev_chain = train_seq.replace(" QS: ", " QI: ") + " A: " + json.dumps(answers)
                curr_assignment["#" + str(idx + 1)] = answers
            train_seqs.append(prev_chain + " QS: [EOQ]")
            decomp = deepcopy(qa_pair)
            decomp["train_seqs"] = train_seqs
            decomp_json.append(decomp)
            if isinstance(last_answer, list):
                pred_json[qid] = last_answer
            else:
                pred_json[qid] = str(last_answer)

    if args.pred_json:
        with open(args.pred_json, "w") as output_fp:
            json.dump(pred_json, output_fp, indent=2)
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
