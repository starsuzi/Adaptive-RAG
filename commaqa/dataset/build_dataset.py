import argparse
import json
import logging
import os
import random
from math import ceil
from random import shuffle
from shutil import copyfile
from typing import List

import _jsonnet

from commaqa.configs.dataset_build_config import DatasetBuildConfig
from commaqa.execution.utils import build_models

logger = logging.getLogger(__name__)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Build a CommaQA dataset from inputs")
    arg_parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Input JSON configuration files " "(comma-separated for multiple files)",
    )
    arg_parser.add_argument("--output", "-o", type=str, required=True, help="Output folder")
    arg_parser.add_argument(
        "--num_groups", type=int, required=False, default=100, help="Number of example groups to create"
    )
    arg_parser.add_argument(
        "--num_examples_per_group", type=int, required=False, default=10, help="Number of examples per group"
    )
    arg_parser.add_argument(
        "--entity_percent",
        type=float,
        required=False,
        default=1.0,
        help="Percentage of entities to sample for each group",
    )
    arg_parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging")

    return arg_parser.parse_args()


class DatasetBuilder:
    def __init__(self, configs: List[DatasetBuildConfig]):
        self.configs = configs

    def build_dataset(self, num_entities_per_group=5, num_groups: int = 10, num_examples_per_group: int = 10):
        data = []
        numqs_per_theory = {}
        logger.info(
            "Creating examples with {} questions per group. #Groups: {}. "
            "#Entities per group: {}".format(num_examples_per_group, num_groups, num_entities_per_group)
        )
        # Distribution over input configs to sample equal #examples from each config
        # Start with equal number (num_groups) and reduce by len(configs) each time
        # This will ensure that we get num_groups/len(configs) groups per config as well as
        # a peaky distribution that samples examples from rarely used configs
        config_distribution = [num_groups for x in self.configs]
        group_idx = 0
        num_attempts = 0
        while group_idx < num_groups:
            num_attempts += 1
            config_idx = random.choices(range(len(self.configs)), config_distribution)[0]
            current_config = self.configs[config_idx]
            # sample entities based on the current config
            entities = current_config.entities.subsample(num_entities_per_group)

            # build a KB based on the entities
            complete_kb = {}
            complete_kb_fact_map = {}
            for pred in current_config.predicates:
                complete_kb[pred.pred_name] = pred.populate_kb(entities)
                curr_pred_kb_fact_map = pred.generate_kb_fact_map(complete_kb)
                for k, v in curr_pred_kb_fact_map.items():
                    complete_kb_fact_map[k] = v

            questions_per_theory = {}
            context = " ".join(complete_kb_fact_map.values())

            output_data = {
                "kb": complete_kb,
                "context": context,
                "per_fact_context": complete_kb_fact_map,
                "pred_lang_config": current_config.pred_lang_config.model_config_as_json(),
            }

            # build questions using KB and language config
            model_library = build_models(current_config.pred_lang_config.model_config, complete_kb)
            for theory in current_config.theories:
                theory_qs = theory.create_questions(
                    entities=entities.entity_type_map,
                    pred_lang_config=current_config.pred_lang_config,
                    model_library=model_library,
                )
                theory_key = theory.to_str()
                if theory_key not in numqs_per_theory:
                    numqs_per_theory[theory_key] = 0
                numqs_per_theory[theory_key] += len(theory_qs)
                questions_per_theory[theory_key] = theory_qs
            all_questions = [qa for qa_per_theory in questions_per_theory.values() for qa in qa_per_theory]
            if len(all_questions) < num_examples_per_group:
                # often happens when a configuration has only one theory, skip print statement
                if len(current_config.theories) != 1:
                    logger.warning(
                        "Insufficient examples: {} generated. Sizes:{} KB:\n{}".format(
                            len(all_questions),
                            [(tidx, len(final_questions)) for (tidx, final_questions) in questions_per_theory.items()],
                            json.dumps(complete_kb, indent=2),
                        )
                    )
                logger.debug("Skipping config: {} Total #questions: {}".format(config_idx, len(all_questions)))
                continue

            # subsample questions to equalize #questions per theory
            min_size = min([len(qa) for qa in questions_per_theory.values()])
            subsampled_questions = []
            for qa_per_theory in questions_per_theory.values():
                subsampled_questions.extend(random.sample(qa_per_theory, min_size))
            if len(subsampled_questions) < num_examples_per_group:
                logger.warning(
                    "Skipping config: {} Sub-sampled questions: {}".format(config_idx, len(subsampled_questions))
                )
                continue
            final_questions = random.sample(subsampled_questions, num_examples_per_group)
            output_data["all_qa"] = all_questions
            output_data["qa_pairs"] = final_questions
            data.append(output_data)
            group_idx += 1
            # update distribution over configs
            config_distribution[config_idx] -= len(self.configs)
            if group_idx % 100 == 0:
                logger.info("Created {} groups. Attempted: {}".format(group_idx, num_attempts))
        for theory_key, numqs in numqs_per_theory.items():
            logger.debug("Theory: <{}> \n NumQs: [{}]".format(theory_key, numqs))
        return data


if __name__ == "__main__":
    args = parse_arguments()
    dataset_configs = []
    counter = 0
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    for filename in args.input_json.split(","):
        counter += 1
        output_dir = ""
        # if output is a json file
        if args.output.endswith(".json"):
            output_dir = os.path.dirname(args.output)
        else:
            output_dir = args.output

        if filename.endswith(".jsonnet"):
            data = json.loads(_jsonnet.evaluate_file(filename))
            # dump the configuration as a the source file
            with open(output_dir + "/source{}.json".format(counter), "w") as output_fp:
                json.dump(data, output_fp, indent=2)
            dataset_config = DatasetBuildConfig(data)
            dataset_configs.append(dataset_config)
        else:
            # dump the configuration as a the source file
            copyfile(filename, output_dir + "/source{}.json".format(counter))
            with open(filename, "r") as input_fp:
                input_json = json.load(input_fp)
                dataset_config = DatasetBuildConfig(input_json)
                dataset_configs.append(dataset_config)

    builder = DatasetBuilder(dataset_configs)
    data = builder.build_dataset(
        num_groups=args.num_groups,
        num_entities_per_group=args.entity_percent,
        num_examples_per_group=args.num_examples_per_group,
    )
    num_examples = len(data)
    print("Number of example groups: {}".format(num_examples))
    if args.output.endswith(".json"):
        print("Single file output name provided (--output file ends with .json)")
        print("Dumping examples into a single file instead of train/dev/test splits")
        with open(args.output, "w") as output_fp:
            json.dump(data, output_fp, indent=4)
    else:
        shuffle(data)
        train_ex = ceil(num_examples * 0.8)
        dev_ex = ceil(num_examples * 0.1)
        test_ex = num_examples - train_ex - dev_ex
        print("Train/Dev/Test: {}/{}/{}".format(train_ex, dev_ex, test_ex))
        files = [args.output + "/train.json", args.output + "/dev.jsonl", args.output + "/test.json"]
        datasets = [data[:train_ex], data[train_ex : train_ex + dev_ex], data[train_ex + dev_ex :]]
        for file, dataset in zip(files, datasets):
            with open(file, "w") as output_fp:
                json.dump(dataset, output_fp, indent=4)
