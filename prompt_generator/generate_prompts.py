import os
import argparse
from typing import List, Dict

from prompt_generator.common import QAPromptGenerator, NoContextOpenRetrieverPromptGenerator


def get_qa_prompt_generator_args_and_names(dataset_name: str) -> List[Dict]:
    max_paragraph_tokens = 250  # keep it fixed 250.
    prompt_generator_args_and_names = []
    model_names = ["codex", "flan_t5"]
    for model_name in model_names:

        for qa_type in ("direct", "cot"):
            for context_type in ("no", "gold_with_distractors"):
                distractor_counts = (0,) if context_type == "no" else (1, 2, 3)
                for distractor_count in distractor_counts:

                    if distractor_count == 0:
                        assert context_type == "no"

                    prompt_generator_args = {
                        "qa_type": qa_type,
                        "context_type": context_type,
                        "distractor_count": distractor_count,
                        "model_name": model_name,
                    }
                    if dataset_name == "iirc" and model_name == "flan_t5":
                        prompt_generator_args["pinned_at_bottom"] = model_name == "flan_t5"

                    context_type_ = f"gold_with_{distractor_count}_distractors"
                    if not distractor_count:
                        context_type_ = "no"

                    prompt_name = f"{context_type_}_context_{qa_type}_qa_{model_name}.txt"
                    prompt_generator_args_and_names.append(
                        {
                            "generator_args": prompt_generator_args,
                            "name": prompt_name,
                            "max_paragraph_tokens": max_paragraph_tokens,
                        }
                    )

    return prompt_generator_args_and_names


def get_no_context_open_retrieval_prompt_generator_args_and_names(dataset_name: str) -> List[Dict]:
    max_paragraph_tokens = 250
    prompt_generator_args_and_names = []

    prompt_name = "no_context_open_llm_retrieval_codex.txt"
    prompt_generator_args_and_names.append(
        {"generator_args": {"model_name": "codex"}, "name": prompt_name, "max_paragraph_tokens": max_paragraph_tokens}
    )

    prompt_name = "no_context_open_llm_retrieval_flan_t5.txt"
    prompt_generator_args_and_names.append(
        {"generator_args": {"model_name": "flan_t5"}, "name": prompt_name, "max_paragraph_tokens": max_paragraph_tokens}
    )

    return prompt_generator_args_and_names


def main():

    parser = argparse.ArgumentParser(description="Generate prompts.")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", choices={"hotpotqa", "2wikimultihopqa", "musique", "iirc"}
    )
    args = parser.parse_args()

    input_file_path = os.path.join("processed_data", args.dataset_name, "annotated_only_train.jsonl")
    output_directory = os.path.join("prompts", args.dataset_name)

    task_names = ["qa"]
    if args.dataset_name == "iirc":
        task_names.append("no_context_open_retrieval")

    for task_name in task_names:

        if task_name == "qa":
            args_name_generator = get_qa_prompt_generator_args_and_names
            prompt_generator_cls = QAPromptGenerator
        elif task_name == "no_context_open_retrieval":
            args_name_generator = get_no_context_open_retrieval_prompt_generator_args_and_names
            prompt_generator_cls = NoContextOpenRetrieverPromptGenerator
        else:
            raise Exception(f"Invalid task_name {task_name}")

        for prompt_args_and_name in args_name_generator(args.dataset_name):

            generator_args = prompt_args_and_name["generator_args"]
            generator_args["input_file_path"] = input_file_path
            prompt_generator = prompt_generator_cls(**generator_args)

            output_file_name = prompt_args_and_name["name"]
            output_file_path = os.path.join(output_directory, output_file_name)

            prompt_args_and_name.pop("generator_args")
            prompt_args_and_name.pop("name")
            prompt_args_and_name.pop("max_paragraph_tokens")
            if prompt_args_and_name:
                raise Exception("Looks like prompt_args_and_name has extra unused args.")

            print(f"Writing in {output_file_path}")
            with open(output_file_path, "w") as file:
                file.write(prompt_generator.generate())


if __name__ == "__main__":
    main()
