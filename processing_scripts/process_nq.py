import os

from lib import read_jsonl, write_jsonl, read_json
import json
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Union, Any

random.seed(13370)  # Don't change.

def safe_sample(items: List[Any], count: int) -> List[Any]:
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []

def write_nq_instances_to_filepath(raw_instances, output_directory: str, set_name: str):

    print(f"Writing in: {output_directory}")
    
    print(len(raw_instances))     
    
    with open(output_directory, "w") as output_file:

        for idx, raw_instance in tqdm(enumerate(raw_instances)):

            # Generic RC Format
            processed_instance = {}
            processed_instance["dataset"] = "nq"
            processed_instance["question_id"] = 'single_nq_'+set_name+'_'+str(idx)
            processed_instance["question_text"] = raw_instance["question"]
            #processed_instance["level"] = raw_instance["level"]
            #processed_instance["type"] = raw_instance["type"]

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": raw_instance["answers"],
            }
            #import pdb; pdb.set_trace()

            processed_instance["answers_objects"] = [answers_object]

            lst_context = []
            context_id = 0

            for ctx in raw_instance['positive_ctxs']:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = True

                context_id = context_id + 1

                lst_context.append(dict_context)

            lst_neg_ctxs = raw_instance['negative_ctxs']
            sampled_lst_neg_ctxs = safe_sample(lst_neg_ctxs, 5)

            for ctx in sampled_lst_neg_ctxs:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = False

                context_id = context_id + 1

                lst_context.append(dict_context)

            lst_hardneg_ctxs = raw_instance['hard_negative_ctxs']
            sampled_lst_hardneg_ctxs = safe_sample(lst_hardneg_ctxs, 5)

            for ctx in sampled_lst_hardneg_ctxs:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = False

                context_id = context_id + 1

                lst_context.append(dict_context)


            processed_instance["contexts"] = lst_context

            output_file.write(json.dumps(processed_instance) + "\n")


if __name__ == "__main__":

    input_directory = os.path.join("raw_data", "nq")
    output_directory = os.path.join("processed_data", "nq")
    os.makedirs(output_directory, exist_ok=True)

    output_filepath = os.path.join(output_directory, "train.jsonl")
    input_filepath = os.path.join(input_directory, f"biencoder-nq-train.json")
    raw_instances = read_json(input_filepath)
    write_nq_instances_to_filepath(raw_instances, output_filepath, 'train')

    output_filepath = os.path.join(output_directory, "dev.jsonl")
    input_filepath = os.path.join(input_directory, f"biencoder-nq-dev.json")
    raw_instances = read_json(input_filepath)
    write_nq_instances_to_filepath(raw_instances, output_filepath, 'dev')