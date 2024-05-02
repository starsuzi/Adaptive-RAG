import json
import os
from preprocess_utils import *

lst_dataset_name = ['musique', '2wikimultihopqa', 'hotpotqa', 'nq', 'trivia', 'squad']

# 2wikimultihopqa
train_input_file = os.path.join("raw_data", '2wikimultihopqa', 'train.json')
train_output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', '2wikimultihopqa_train.json')
save_inductive_bias_2wikimultihopqa(train_input_file, train_output_file)

# hotpotqa
train_input_file = os.path.join("raw_data", 'hotpotqa', 'hotpot_train_v1.1.json')
train_output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', 'hotpotqa_train.json')
save_inductive_bias_hotpotqa(train_input_file, train_output_file)

# musique
train_input_file = os.path.join("raw_data", 'musique', 'musique_ans_v1.0_train.jsonl')
train_output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', 'musique_train.json')
save_inductive_bias_musique(train_input_file, train_output_file)

# nq
single_data_name = 'nq'
train_input_file = os.path.join("raw_data", single_data_name, 'biencoder-nq-train.json')
train_output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', f'{single_data_name}_train.json')
save_inductive_bias_single_data(train_input_file, train_output_file, single_data_name, 'train')

# squad
single_data_name = 'squad'
train_input_file = os.path.join("raw_data", single_data_name, 'biencoder-squad1-train.json')
train_output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', f'{single_data_name}_train.json')
save_inductive_bias_single_data(train_input_file, train_output_file, single_data_name, 'train')

# trivia
single_data_name = 'trivia'
train_input_file = os.path.join("raw_data", single_data_name, 'biencoder-trivia-train.json')
train_output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', f'{single_data_name}_train.json')
save_inductive_bias_single_data(train_input_file, train_output_file, single_data_name, 'train')

# concat total datasets
lst_total_data_per_set = []
for dataset_name in lst_dataset_name:
    input_file = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd', 'binary', f'{dataset_name}_train.json')
    json_data = load_json(input_file)[:400]
    lst_total_data_per_set = lst_total_data_per_set + json_data

output_file = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd', 'binary', 'total_data_train.json')
save_json(output_file, lst_total_data_per_set)