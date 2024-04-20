import json
import jsonlines
import os
from preprocess_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="model name.", choices=("flan_t5_xl", "flan_t5_xxl", "gpt"))
args = parser.parse_args()

# Set your file path accordingly
orig_nq_file = os.path.join("processed_data", "nq", 'dev_500_subsampled.jsonl')
orig_trivia_file = os.path.join("processed_data", "trivia", 'dev_500_subsampled.jsonl')
orig_squad_file = os.path.join("processed_data", "squad", 'dev_500_subsampled.jsonl')
orig_musique_file = os.path.join("processed_data", "musique", 'dev_500_subsampled.jsonl')
orig_hotpotqa_file = os.path.join("processed_data", "hotpotqa", 'dev_500_subsampled.jsonl')
orig_wikimultihopqa_file = os.path.join("processed_data", "2wikimultihopqa", 'dev_500_subsampled.jsonl')

nq_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json') 
trivia_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
squad_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
musique_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
hotpotqa_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
wikimultihopqa_multi_file = os.path.join("predictions", "dev_500", f'ircot_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')

nq_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json') 
trivia_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_trivia____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
squad_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_squad____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
musique_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
hotpotqa_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
wikimultihopqa_one_file = os.path.join("predictions", "dev_500", f'oner_qa_{args.model_name}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')

nq_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_nq____prompt_set_1', 'zero_single_multi_classification__nq_to_nq__dev_500_subsampled.json') 
trivia_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_trivia____prompt_set_1', 'zero_single_multi_classification__trivia_to_trivia__dev_500_subsampled.json')
squad_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_squad____prompt_set_1', 'zero_single_multi_classification__squad_to_squad__dev_500_subsampled.json')
musique_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_musique____prompt_set_1', 'zero_single_multi_classification__musique_to_musique__dev_500_subsampled.json')
hotpotqa_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_hotpotqa____prompt_set_1', 'zero_single_multi_classification__hotpotqa_to_hotpotqa__dev_500_subsampled.json')
wikimultihopqa_zero_file = os.path.join("predictions", "dev_500", f'nor_qa_{args.model_name}_2wikimultihopqa____prompt_set_1', 'zero_single_multi_classification__2wikimultihopqa_to_2wikimultihopqa__dev_500_subsampled.json')

output_path = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd', args.model_name, 'silver')
    
lst_nq = label_complexity(orig_nq_file, nq_zero_file, nq_one_file, nq_multi_file, 'nq')
lst_trivia = label_complexity(orig_trivia_file, trivia_zero_file, trivia_one_file, trivia_multi_file, 'trivia')
lst_squad = label_complexity(orig_squad_file, squad_zero_file, squad_one_file, squad_multi_file, 'squad')
lst_musique = label_complexity(orig_musique_file, musique_zero_file, musique_one_file, musique_multi_file, 'musique')
lst_hotpotqa = label_complexity(orig_hotpotqa_file, hotpotqa_zero_file, hotpotqa_one_file, hotpotqa_multi_file, 'hotpotqa')
lst_wikimultihopqa = label_complexity(orig_wikimultihopqa_file, wikimultihopqa_zero_file, wikimultihopqa_one_file, wikimultihopqa_multi_file, '2wikimultihopqa')


lst_total_data = lst_musique + lst_hotpotqa + lst_wikimultihopqa + lst_nq + lst_trivia + lst_squad


save_json(output_path+'/train.json', lst_total_data)



