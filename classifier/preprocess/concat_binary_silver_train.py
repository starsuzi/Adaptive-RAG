import json, os
from preprocess_utils import *
import sys

lst_total = []

lst_dataset_name = ['musique', '2wikimultihopqa', 'hotpotqa', 'nq', 'trivia', 'squad']
lst_set_name = ['train', 'valid']
lst_model_name = ['flan_t5_xl', 'flan_t5_xxl', 'gpt']

# train
for model_name in lst_model_name:
    binary_input_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', 'total_data_train.json')
    silver_input_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", model_name, 'silver', 'train.json')
    output_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", model_name, 'binary_silver', 'train.json')
    concat_and_save_binary_silver(binary_input_file, silver_input_file, output_file)


# # valid
# min_len = get_binary_min_len(lst_dataset_name)
# for model_name in lst_model_name:
#     binary_input_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", 'binary', 'total_data_valid.json')
#     silver_valid_input_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", model_name, 'valid.json')
#     output_valid_file = os.path.join('classifier', "data", "musique_hotpot_wiki2_nq_tqa_sqd", model_name, 'binary_silver', 'valid.json')
#     concat_and_save_binary_silver(binary_input_file, silver_valid_input_file, output_valid_file, min_len=min_len)
