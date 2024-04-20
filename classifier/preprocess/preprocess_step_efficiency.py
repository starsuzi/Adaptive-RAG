import json
import jsonlines
from preprocess_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name.", choices=("flan_t5_xl", "flan_t5_xxl", "gpt"))
args = parser.parse_args()

output_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}', 'total', 'stepNum.json')

musique_multi_step_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__musique_to_musique__test_subsampled_chains.txt')
hotpotqa_multi_step_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled_chains.txt')
wikimultihopqa_multi_step_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled_chains.txt')
nq_multi_step_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__nq_to_nq__test_subsampled_chains.txt') 
trivia_multi_step_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__trivia_to_trivia__test_subsampled_chains.txt')
squad_multi_step_file = os.path.join("predictions", "test", f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__squad_to_squad__test_subsampled_chains.txt')


musique_dict_qid_to_stepNum = count_stepNum(musique_multi_step_file)
hotpotqa_dict_qid_to_stepNum = count_stepNum(hotpotqa_multi_step_file)
wikimultihopqa_dict_qid_to_stepNum = count_stepNum(wikimultihopqa_multi_step_file)
nq_dict_qid_to_stepNum = count_stepNum(nq_multi_step_file)
trivia_dict_qid_to_stepNum = count_stepNum(trivia_multi_step_file)
squad_dict_qid_to_stepNum = count_stepNum(squad_multi_step_file)

total_dict_qid_to_stepNum = dict()
total_dict_qid_to_stepNum.update(musique_dict_qid_to_stepNum)
total_dict_qid_to_stepNum.update(hotpotqa_dict_qid_to_stepNum)
total_dict_qid_to_stepNum.update(wikimultihopqa_dict_qid_to_stepNum)
total_dict_qid_to_stepNum.update(nq_dict_qid_to_stepNum)
total_dict_qid_to_stepNum.update(trivia_dict_qid_to_stepNum)
total_dict_qid_to_stepNum.update(squad_dict_qid_to_stepNum)

save_json(output_file, total_dict_qid_to_stepNum)
