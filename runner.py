import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Wrapper around run.py to make experimentation easier.")
    parser.add_argument("system", type=str, choices=("ircot", "ircot_qa", "oner", "oner_qa", "nor_qa"))
    parser.add_argument("model", type=str, choices=("flan-t5-xxl", "flan-t5-xl", "none", 'gpt'))
    all_datasets = ["hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad']
    all_datasets += ["_to_".join([dataset_a, dataset_b]) for dataset_a in all_datasets for dataset_b in all_datasets]
    parser.add_argument("dataset", type=str, choices=all_datasets)
    parser.add_argument(
        "command",
        type=str,
        help="command",
        choices={
            "print",
            "write",
            "verify",
            "predict",
            "evaluate",
            "track",
            "summarize",
            "ground_truth_check",
            "backup",
            "print_backup",
            "recover_backup",
            "delete_predictions",
        },
    )
    # TODO
    # llm_port_num
    parser.add_argument(
        "--llm_port_num",
        type=str,
        help="llm_port_num",
        default="8010",
    )
    parser.add_argument(
        "--prompt_set",
        type=str,
        help="prompt_set",
        choices={"1", "2", "3", "aggregate"},
        default="1",
    )
    parser.add_argument("--dry_run", action="store_true", default=False, help="dry_run")
    parser.add_argument("--use_backup", action="store_true", default=False, help="pass --use_backup flag")
    parser.add_argument("--skip_evaluation_path", action="store_true", default=False, help="skip_evaluation_path")
    parser.add_argument("--eval_test", action="store_true", default=False, help="eval_test")
    parser.add_argument("--sample_size", type=int, help="sample_size")
    parser.add_argument("--best", action="store_true", default=False, help="pass --best flag")
    parser.add_argument("--skip_if_exists", action="store_true", default=False, help="skip evaluation of it exists.")
    parser.add_argument(
        "--only_print", action="store_true", default=False, help="print only for eval, ignore otherwise."
    )
    parser.add_argument("--force", action="store_true", default=False, help="force predict if it exists")
    parser.add_argument(
        "--official", action="store_true", default=False, help="use official evaluation for evaluate and summarize."
    )
    args = parser.parse_args()

    if "_to_" in args.dataset:
        train_dataset, eval_dataset = args.dataset.split("_to_")
    else:
        train_dataset = eval_dataset = args.dataset

    experiment_name = "_".join([args.system, args.model.replace("-", "_"), args.dataset])
    if args.model == "none":
        experiment_name = "_".join([args.system, args.dataset])
    instantiation_scheme = args.system


    set_name = "test" if args.eval_test else 'dev_' + str(args.sample_size)
    run_command_array = [
        f"python run.py {args.command} {experiment_name} --instantiation_scheme {instantiation_scheme} --prompt_set {args.prompt_set} --set_name {set_name} --llm_port_num {args.llm_port_num}",
    ]

    if args.command in ("write", "predict", "evaluate", "print", "summarize") and args.best:
        run_command_array += ["--best"]

    if args.command == "write":
        run_command_array += ["--no_diff"]

    if (
        args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check")
        and not args.skip_evaluation_path
    ) or args.best:
        set_name = "test" if args.eval_test else 'dev_' + str(args.sample_size)
        evaluation_path = os.path.join("processed_data", eval_dataset, f"{set_name}_subsampled.jsonl")
        run_command_array += [f"--evaluation_path {evaluation_path}"]

    if (
        args.command in ("predict", "summarize") or (args.command == "write" and args.best)
    ) and train_dataset != eval_dataset:
        variable_replacements = {"retrieval_corpus_name": f'"{eval_dataset}"'}
        variable_replacements_str = json.dumps(variable_replacements).replace(" ", "")
        run_command_array += ["--variable_replacements", f"'{variable_replacements_str}'"]

    if args.command in ("predict"):
        run_command_array.append("--skip_if_exists --silent")

    if args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check") and args.use_backup:
        run_command_array += ["--use_backup"]

    if args.command == "predict" and args.force:
        run_command_array += ["--force"]

    if args.command == "evaluate" and args.skip_if_exists:
        run_command_array += ["--skip_if_exists"]

    if args.command == "evaluate" and args.only_print:
        run_command_array += ["--only_print"]

    if args.command in ("evaluate", "summarize") and args.official:
        run_command_array += ["--official"]

    assert train_dataset in experiment_name

    print("", flush=True)
    message = f"Experiment Name: {experiment_name}"
    print("*" * len(message), flush=True)
    print(message, flush=True)
    print("*" * len(message), flush=True)

    run_command_str = " ".join(run_command_array)
    print(run_command_str + "\n", flush=True)
    if not args.dry_run:
        subprocess.call(run_command_str, shell=True)


if __name__ == "__main__":
    main()
