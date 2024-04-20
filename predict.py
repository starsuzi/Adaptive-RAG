import os
import shutil
import subprocess
import argparse

from lib import (
    get_retriever_address,
    get_llm_server_address,
    infer_source_target_prefix,
    get_config_file_path_from_name_or_path,
)


def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main():
    parser = argparse.ArgumentParser(description="Run configurable_inference on given config and dataset.")
    parser.add_argument("experiment_name_or_path", type=str, help="experiment_name_or_path")
    parser.add_argument("evaluation_path", type=str, help="evaluation_path")
    parser.add_argument(
        "--prediction-suffix", type=str, help="optional suffix for the prediction directory.", default=""
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="dry-run")
    parser.add_argument("--skip-evaluation", type=str, default="", help="skip-evaluation")
    parser.add_argument("--force", action="store_true", default=False, help="force predict if it exists")
    parser.add_argument(
        "--variable-replacements",
        type=str,
        help="json string for jsonnet local variable replacements.",
        default="",
    )
    parser.add_argument("--silent", action="store_true", help="silent")
    parser.add_argument('--set_name', type=str, help="set_name", required=True)
    # TODO
    # llm_port_num
    parser.add_argument(
        '--llm_port_num', type=str, help="llm_port_num", required=True
    )
    args = parser.parse_args()

    config_filepath = get_config_file_path_from_name_or_path(args.experiment_name_or_path)
    experiment_name = os.path.splitext(os.path.basename(config_filepath))[0]
    prediction_directory = os.path.join("predictions", args.set_name, experiment_name + args.prediction_suffix)

    os.makedirs(prediction_directory, exist_ok=True)

    prediction_filename = os.path.splitext(os.path.basename(args.evaluation_path))[0]
    prediction_filename = infer_source_target_prefix(config_filepath, args.evaluation_path) + prediction_filename
    prediction_filepath = os.path.join(prediction_directory, "prediction__" + prediction_filename + ".json")

    if os.path.exists(prediction_filepath) and not args.force:
        from run import is_experiment_complete

        metrics_file_path = os.path.join(prediction_directory, "evaluation_metrics__" + prediction_filename + ".json")
        if is_experiment_complete(config_filepath, prediction_filepath, metrics_file_path, args.variable_replacements):
            exit(f"The prediction_file_path {prediction_filepath} already exists and is complete. Pass --force.")

    env_variables = {}
    retriever_address = get_retriever_address()
    env_variables["RETRIEVER_HOST"] = str(retriever_address["host"])
    env_variables["RETRIEVER_PORT"] = str(retriever_address["port"])
    llm_server_address = get_llm_server_address(args.llm_port_num)
    env_variables["LLM_SERVER_HOST"] = str(llm_server_address["host"])
    env_variables["LLM_SERVER_PORT"] = str(llm_server_address["port"])

    env_variables_str = " ".join([f"{key}={value}" for key, value in env_variables.items()]).strip()

    predict_command = " ".join(
        [
            env_variables_str,
            "python -m commaqa.inference.configurable_inference",
            f"--config {config_filepath}",
            f"--input {args.evaluation_path}",
            f"--output {prediction_filepath}",
        ]
    ).strip()

    if args.silent:
        predict_command += " --silent"

    if args.variable_replacements:
        predict_command += f" --variable-replacements '{args.variable_replacements}'"

    print(f"Run predict_command: \n{predict_command}\n")

    if not args.dry_run:
        subprocess.call(predict_command, shell=True)

    # To be able to reproduce the same result:
    git_hash_filepath = os.path.join(prediction_directory, "git_hash__" + prediction_filename + ".txt")

    # Again for reproducibility:
    backup_config_filepath = os.path.join(prediction_directory, "config__" + prediction_filename + ".jsonnet")
    shutil.copyfile(config_filepath, backup_config_filepath)

    if not args.skip_evaluation:
        
        evaluate_command = " ".join(["python evaluate.py", str(config_filepath), str(args.evaluation_path), '--set_name', args.set_name, '--llm_port_num', args.llm_port_num]).strip()

        print(f"Run evaluate_command: \n{evaluate_command}\n")

        if not args.dry_run:
            subprocess.call(evaluate_command, shell=True)
        
        evaluate_command = " ".join(
            ["python evaluate.py", str(config_filepath), str(args.evaluation_path), "--official", '--set_name', args.set_name, '--llm_port_num', args.llm_port_num]
        ).strip()

        print(f"Run evaluate_command: \n{evaluate_command}\n")

        if not args.dry_run:
            subprocess.call(evaluate_command, shell=True)


if __name__ == "__main__":
    main()
