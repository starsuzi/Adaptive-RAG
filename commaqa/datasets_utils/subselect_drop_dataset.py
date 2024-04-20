import argparse
import json
import random


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Convert HotPotQA dataset into SQUAD format")
    arg_parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    arg_parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    arg_parser.add_argument(
        "--atype",
        type=str,
        action="append",
        choices=["SPAN", "SPANS", "DATE", "NUMBER"],
        help="Select questions with answers of this type.",
    )
    arg_parser.add_argument(
        "--nth",
        type=int,
        required=False,
        action="append",
        help="Select questions from the nth (of k) paras." " Can provide multiple values.",
    )
    arg_parser.add_argument(
        "--k", type=int, required=False, default=1, help="Select questions from the nth (of k) paras"
    )
    arg_parser.add_argument(
        "--qprob", type=float, required=False, default=1.0, help="Select questions with this probability"
    )
    arg_parser.add_argument("--maxq", type=int, required=False, default=-1, help="Select upto these many questions")
    arg_parser.add_argument("--ids", type=str, required=False, help="File of question ids to select")
    arg_parser.add_argument("--incl_para", type=str, required=False, help="File of para ids that must be included")
    arg_parser.add_argument("--excl_para", type=str, required=False, help="File of para ids that must be excluded")
    return arg_parser.parse_args()


def get_answer_types(answer_json):
    atypes = []

    if len(answer_json["spans"]) > 1:
        atypes.append("SPANS")
    elif len(answer_json["spans"]) == 1:
        atypes.append("SPAN")
    elif answer_json["number"] != "":
        atypes.append("NUMBER")
    else:
        date_json = answer_json["date"]
        if not (date_json["day"] or date_json["month"] or date_json["year"]):
            print("Number, Span or Date not set in {}".format(answer_json))
        else:
            atypes.append("DATE")
    return atypes


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.input, "r") as input_fp:
        input_json = json.load(input_fp)
    ids = None
    incl_para = None
    excl_para = None
    if args.ids:
        with open(args.ids, "r") as input_fp:
            lines = input_fp.read()
        ids = set(lines.split("\n"))
    if args.incl_para:
        with open(args.incl_para, "r") as input_fp:
            lines = input_fp.read()
        incl_para = set(lines.split("\n"))
    if args.excl_para:
        with open(args.excl_para, "r") as input_fp:
            lines = input_fp.read()
        excl_para = set(lines.split("\n"))
    para_counter = 0
    num_questions = 0
    output_json = {}

    for paraid, item in input_json.items():
        para_counter += 1

        if args.nth is not None and para_counter % args.k not in args.nth:
            # if these question should be included, dont skip
            if not (incl_para is not None and paraid in incl_para):
                continue
        if excl_para is not None and paraid in excl_para:
            continue
        para = item["passage"]
        out_pairs = []
        for qa_pair in item["qa_pairs"]:
            if num_questions >= args.maxq > 0:
                break
            question = qa_pair["question"]
            qid = qa_pair["query_id"]
            if ids is not None:
                if qid not in ids:
                    continue

            a_accept = True

            if args.atype is not None:
                a_accept = False
                atypes = get_answer_types(qa_pair["answer"])
                if len(atypes) == 0:
                    # no matching answer type
                    print(qid, question)
                for atype in atypes:
                    if atype in args.atype:
                        a_accept = True
                        break

            if a_accept and random.random() < args.qprob:
                out_pairs.append(qa_pair)
                num_questions += 1

        if len(out_pairs):
            item["qa_pairs"] = out_pairs
            output_json[paraid] = item

    with open(args.output, "w") as output_fp:
        json.dump(output_json, output_fp, indent=2)
    print("Num output questions: {}".format(num_questions))
