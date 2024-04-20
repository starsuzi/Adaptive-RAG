import argparse
import json
import logging
import time
import os

from tqdm import tqdm
import _jsonnet

from commaqa.inference.constants import MODEL_NAME_CLASS, READER_NAME_CLASS
from commaqa.inference.dataset_readers import DatasetReader
from commaqa.inference.model_search import ModelController, BestFirstDecomposer
from commaqa.inference.data_instances import StructuredDataInstance
from commaqa.inference.participant_execution import ExecutionParticipant
from commaqa.inference.utils import get_environment_variables

logger = logging.getLogger(__name__)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Convert HotPotQA dataset into SQUAD format")
    arg_parser.add_argument("--input", type=str, required=False, help="Input QA file")
    arg_parser.add_argument("--output", type=str, required=False, help="Output file")
    arg_parser.add_argument("--config", type=str, required=True, help="Model configs")
    arg_parser.add_argument(
        "--reader", type=str, required=False, help="Dataset reader", choices=READER_NAME_CLASS.keys()
    )
    arg_parser.add_argument("--debug", action="store_true", default=False, help="Debug output")
    arg_parser.add_argument("--silent", action="store_true", default=False, help="Silent / no prints")
    arg_parser.add_argument("--demo", action="store_true", default=False, help="Demo mode")
    arg_parser.add_argument("--threads", default=1, type=int, help="Number of threads (use MP if set to >1)")
    arg_parser.add_argument("--openai-api-keyname", type=str, help="optional OpenAI API keyname.", default=None)
    arg_parser.add_argument("--llm-server-key-suffix", type=str, help="optional LLM server key suffix.", default="")
    arg_parser.add_argument(
        "--variable-replacements",
        type=str,
        help="json string for jsonnet local variable replacements.",
        default="",
    )
    return arg_parser.parse_args()


def build_decomposer_and_models(config_map):
    # print("loading participant models (might take a while)...")
    model_map = {}
    for key, value in config_map["models"].items():
        class_name = value.pop("name")
        if class_name not in MODEL_NAME_CLASS:
            raise ValueError(
                "No class mapped to model name: {} in MODEL_NAME_CLASS:{}".format(class_name, MODEL_NAME_CLASS)
            )
        model = MODEL_NAME_CLASS[class_name](**value)
        if key in config_map:
            raise ValueError(
                "Overriding key: {} with value: {} using instantiated model of type:"
                " {}".format(key, config_map[key], class_name)
            )
        config_map[key] = model.query
        model_map[key] = model

    # Special case for ExecutionParticipant
    for model in model_map.values():
        if isinstance(model, ExecutionParticipant):
            model.set_model_lib(model_map)

    ## instantiating
    controller = ModelController(config_map, data_class=StructuredDataInstance)
    decomposer = BestFirstDecomposer(controller)
    return decomposer, model_map


def load_config(config_file):
    if config_file.endswith(".jsonnet"):
        ext_vars = get_environment_variables()
        # logger.info("Parsing config with external variables: {}".format(ext_vars))

        # HACKY: I've to ovvride this way instead of passing the env variables for now.
        if parsed_args.variable_replacements:
            import uuid
            from hp_manager import instatiate_config

            variable_replacements = json.loads(parsed_args.variable_replacements)
            with open(config_file, "r") as file:
                file_content = file.read()
            file_content = instatiate_config(content=file_content, variable_replacements=variable_replacements)
            temp_config_file = os.path.join(os.path.split(config_file)[0], uuid.uuid4().hex)
            assert not os.path.exists(temp_config_file)
            with open(temp_config_file, "w") as file:
                file.write(file_content)

            config_map = json.loads(_jsonnet.evaluate_file(temp_config_file, ext_vars=ext_vars))
            os.remove(temp_config_file)

        else:
            #import pdb; pdb.set_trace()
            config_map = json.loads(_jsonnet.evaluate_file(config_file, ext_vars=ext_vars))

    else:
        if parsed_args.variable_replacements:
            raise Exception("Variable replacement is not possible for json files. It's only applicable for jsonnet.")

        with open(config_file, "r") as input_fp:
            config_map = json.load(input_fp)
    return config_map


def load_reader(args, config_map):
    if "reader" in config_map:
        reader_config = config_map["reader"]
        reader_name = reader_config.pop("name")
        reader: DatasetReader = READER_NAME_CLASS[reader_name](**reader_config)
    else:
        reader: DatasetReader = READER_NAME_CLASS[args.example_reader]()
    return reader


def demo_mode(args, reader, decomposer):
    qid_example_map = {}
    if args.input:
        for eg in reader.read_examples(args.input):
            qid = eg["qid"]
            question = eg["query"]
            output_eg = {"qid": qid, "query": question, "question": question}
            if "paras" in eg:
                output_eg["paras"] = eg["paras"]
            qid_example_map[qid] = output_eg
    while True:
        qid = input("QID: ")
        if qid in qid_example_map:
            example = qid_example_map[qid]
            print("using example from input file: " + json.dumps(example, indent=2))
        else:
            question = input("Question: ")
            example = {"qid": qid, "query": question, "question": question}
        final_state, other_states = decomposer.find_answer_decomp(example, debug=args.debug)
        if final_state is None:
            print("FAILED!")
        else:
            if args.debug:
                for other_state in other_states:
                    data = other_state.data
                    print(data.get_printable_reasoning_chain())
                    print("Score: " + str(other_state._score))
            data = final_state._data
            chain = example["question"]
            chain += "\n" + data.get_printable_reasoning_chain()
            chain += " S: " + str(final_state._score)
            print(chain)


def inference_mode(args, reader, decomposer, model_map, override_answer_by=None):
    print("Running inference on examples")
    qid_answer_chains = []

    start_time = time.time()

    if not args.input:
        raise ValueError("Input file must be specified when run in non-demo mode")
    if args.threads > 1:
        import multiprocessing as mp

        mp.set_start_method("spawn")
        with mp.Pool(args.threads) as p:
            qid_answer_chains = p.map(decomposer.return_qid_prediction, reader.read_examples(args.input))
    else:
        iterator = reader.read_examples(args.input)
        if args.silent:
            iterator = tqdm(iterator)
        for example in iterator:
            qid_answer_chains.append(
                decomposer.return_qid_prediction(
                    example, override_answer_by=override_answer_by, debug=args.debug, silent=args.silent
                )
            )

    end_time = time.time()
    seconds_taken = round(end_time - start_time)

    predictions = {x[0]: x[1] for x in qid_answer_chains}
    print(f"Writing predictions in {args.output}")
    with open(args.output, "w") as output_fp:
        json.dump(predictions, output_fp, indent=4)

    ext_index = args.output.rfind(".")

    time_taken_file_path = args.output[:ext_index] + "_time_taken.txt"
    with open(time_taken_file_path, "w") as file:
        file.write(str(seconds_taken))

    chains = [x[2] for x in qid_answer_chains]
    chain_tsv = args.output[:ext_index] + "_chains.txt"
    with open(chain_tsv, "w") as output_fp:
        for chain in chains:
            output_fp.write(chain + "\n")

    # Also save original full evaluation path.
    full_eval_path = args.output[:ext_index] + "_full_eval_path.txt"
    with open(full_eval_path, "w") as output_fp:
        output_fp.write(args.input)

    # Also save variable_replacements to assure same exps don't conflict.
    # Update experiment completioon to deal with this.
    variable_replacements_path = args.output[:ext_index] + "_variable_replacements.json"
    with open(variable_replacements_path, "w") as output_fp:
        output_fp.write(args.variable_replacements)


if __name__ == "__main__":

    parsed_args = parse_arguments()
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.ERROR)

    if parsed_args.openai_api_keyname is not None:
        assert os.environ.get(
            parsed_args.openai_api_keyname, ""
        ), f"No API key found in the env variable named: {parsed_args.openai_api_keyname}"
        os.environ["OPENAI_API_KEY"] = os.environ[parsed_args.openai_api_keyname]

        # For some reason the following is needed inspite of the setting ENV variable above.
        import openai  # import only if needed

        openai.api_key = os.environ[parsed_args.openai_api_keyname]

        print(f"OpenAI API key name used: {parsed_args.openai_api_keyname}")

    if parsed_args.llm_server_key_suffix:
        os.environ["LLM_SERVER_KEY_SUFFIX"] = parsed_args.llm_server_key_suffix

    config_map = load_config(parsed_args.config)

    decomposer, model_map = build_decomposer_and_models(config_map)

    example_reader = load_reader(args=parsed_args, config_map=config_map)

    if parsed_args.demo:
        demo_mode(args=parsed_args, reader=example_reader, decomposer=decomposer)
    else:
        override_answer_by = config_map.get("override_answer_by", None)
        inference_mode(
            args=parsed_args,
            reader=example_reader,
            decomposer=decomposer,
            model_map=model_map,
            override_answer_by=override_answer_by,
        )
