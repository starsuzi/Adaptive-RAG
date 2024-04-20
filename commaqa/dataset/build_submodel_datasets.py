import argparse
import json
import os
import random
import re
import string
from math import ceil
from pathlib import Path
from shutil import copyfile
from typing import List

import _jsonnet
from tqdm import tqdm

from commaqa.configs.dataset_build_config import DatasetBuildConfig
from commaqa.dataset.utils import get_predicate_args, dict_product, nonempty_answer
from commaqa.execution.utils import build_models


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Build a ReModeL dataset from inputs")
    arg_parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON configuration files " "(comma-separated for multiple files)",
    )
    arg_parser.add_argument("--output", "-o", type=str, required=True, help="Output folder")
    arg_parser.add_argument(
        "--num_groups", type=int, required=False, default=500, help="Number of example groups to create"
    )
    arg_parser.add_argument(
        "--num_examples_per_group", type=int, required=False, default=10, help="Number of examples per group"
    )
    arg_parser.add_argument(
        "--entity_percent",
        type=float,
        required=False,
        default=0.25,
        help="Percentage of entities to sample for each group",
    )

    return arg_parser.parse_args()


class SubDatasetBuilder:
    def __init__(self, configs: List[DatasetBuildConfig]):
        self.configs = configs

    def build_entities(self, entities, ent_type):
        m = re.match("list\((.*)\)", ent_type)
        if m:
            # too many possible permutations, only build a list of size ent_type*2
            returned_list = []
            ent_type = m.group(1)
            for i in range(len(entities[ent_type]) * 2):
                sample_size = random.choice(range(2, 5))
                sampled_ents = random.sample(entities[ent_type], sample_size)
                returned_list.append(json.dumps(sampled_ents))
            return returned_list
        else:
            return entities[ent_type]

    def build_sub_dataset(self, num_entities_per_group=5, num_groups: int = 10, num_examples_per_group: int = 10):
        per_model_dataset = {}
        for g in tqdm(range(num_groups)):
            config = random.choice(self.configs)
            entities = config.entities.subsample(num_entities_per_group)
            complete_kb = {}
            for pred in config.predicates:
                complete_kb[pred.pred_name] = pred.populate_kb(entities)

            model_library = build_models(config.pred_lang_config.model_config, complete_kb)
            # per_model_qa = {}
            # per_model_kb = {}
            for model, model_configs in config.pred_lang_config.model_config.items():
                all_qa = {}
                gold_kb = {}
                for model_config in model_configs:
                    if model_config.init is None:
                        raise ValueError(
                            "Initialization needs to be specified to build the "
                            "sub-model dataset for {}".format(model_config)
                        )

                    # Add the model-specific kb based on the steps
                    for step in model_config.steps:
                        qpred, qargs = get_predicate_args(step.question)
                        if qpred not in gold_kb:
                            gold_kb[qpred] = complete_kb[qpred]
                    context = ""
                    gold_context = ""
                    for pred in config.predicates:
                        context_rep = pred.generate_context(complete_kb)
                        context += context_rep
                        if pred.pred_name in gold_kb:
                            gold_context += context_rep
                    output_data = {
                        "all_kb": complete_kb,
                        "kb": gold_kb,
                        "context": gold_context,
                        "all_context": context,
                    }
                    # Generate questions
                    assignment_dict = {}
                    # Initialize question arguments
                    for key, ent_type in model_config.init.items():
                        assignment_dict[key] = self.build_entities(entities, ent_type)
                    # For each assignment, generate a question
                    for assignment in dict_product(assignment_dict):
                        if isinstance(model_config.questions, str):
                            questions = [model_config.questions]
                        else:
                            questions = model_config.questions
                        # for each question format, generate a question
                        for question in questions:
                            source_question = question
                            for key, val in assignment.items():
                                question = question.replace(key, val)
                            answers, facts_used = model_library[model].ask_question(question, context)
                            if nonempty_answer(answers):
                                if source_question not in all_qa:
                                    all_qa[source_question] = []
                                all_qa[source_question].append(
                                    {
                                        "question": question,
                                        "answer": answers,
                                        "facts_used": facts_used,
                                        "assignment": assignment,
                                        "config": model_config.to_json(),
                                        "id": "".join([random.choice(string.hexdigits) for n in range(16)]).lower(),
                                    }
                                )
                # subsample questions to equalize #questions per theory
                min_size = min([len(qa) for qa in all_qa.values()])
                subsampled_questions = []
                for qa_per_sourceq in all_qa.values():
                    subsampled_questions.extend(random.sample(qa_per_sourceq, min_size))
                qa = random.sample(subsampled_questions, num_examples_per_group)
                output_data["all_qa"] = [qa for qa_per_sourceq in all_qa.values() for qa in qa_per_sourceq]
                output_data["qa_pairs"] = qa
                if model not in per_model_dataset:
                    per_model_dataset[model] = []
                per_model_dataset[model].append(output_data)
        return per_model_dataset


if __name__ == "__main__":
    args = parse_arguments()
    dataset_configs = []
    counter = 0

    for filename in args.input_json.split(","):
        counter += 1
        output_dir = ""
        if args.output.endswith(".json"):
            output_dir = os.path.dirname(args.output)
        else:
            output_dir = args.output

        if filename.endswith(".jsonnet"):
            data = json.loads(_jsonnet.evaluate_file(filename))
            with open(output_dir + "/source{}.json".format(counter), "w") as output_fp:
                json.dump(data, output_fp, indent=2)
            dataset_config = DatasetBuildConfig(data)
            dataset_configs.append(dataset_config)
        else:
            copyfile(filename, output_dir + "/source{}.json".format(counter))
            with open(filename, "r") as input_fp:
                input_json = json.load(input_fp)
                dataset_config = DatasetBuildConfig(input_json)
                dataset_configs.append(dataset_config)

    builder = SubDatasetBuilder(dataset_configs)
    per_model_dataset = builder.build_sub_dataset(
        num_groups=args.num_groups,
        num_entities_per_group=args.entity_percent,
        num_examples_per_group=args.num_examples_per_group,
    )
    for model, data in per_model_dataset.items():
        num_examples = len(data)
        print("Model: {}".format(model))
        print("Number of example groups: {}".format(num_examples))
        train_ex = ceil(num_examples * 0.8)
        dev_ex = ceil(num_examples * 0.1)
        test_ex = num_examples - train_ex - dev_ex
        print("Train/Dev/Test: {}/{}/{}".format(train_ex, dev_ex, test_ex))
        output_dir = args.output + "/" + model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        files = [output_dir + "/train.json", output_dir + "/dev.jsonl", output_dir + "/test.json"]
        datasets = [data[:train_ex], data[train_ex : train_ex + dev_ex], data[train_ex + dev_ex :]]
        for file, dataset in zip(files, datasets):
            with open(file, "w") as output_fp:
                json.dump(dataset, output_fp, indent=4)
