import json
import jsonlines
from preprocess_utils import *


orig_nq_file = os.path.join("processed_data", "nq", 'test_subsampled.jsonl')
orig_trivia_file = os.path.join("processed_data", "trivia", 'test_subsampled.jsonl')
orig_squad_file = os.path.join("processed_data", "squad", 'test_subsampled.jsonl')
orig_musique_file = os.path.join("processed_data", "musique", 'test_subsampled.jsonl')
orig_hotpotqa_file = os.path.join("processed_data", "hotpotqa", 'test_subsampled.jsonl')
orig_wikimultihopqa_file = os.path.join("processed_data", "2wikimultihopqa", 'test_subsampled.jsonl')
    
lst_musique = prepare_predict_file(orig_musique_file, 'musique')
lst_hotpotqa = prepare_predict_file(orig_hotpotqa_file, 'hotpotqa')
lst_wikimultihopqa = prepare_predict_file(orig_wikimultihopqa_file, '2wikimultihopqa')
lst_nq = prepare_predict_file(orig_nq_file, 'nq')
lst_trivia = prepare_predict_file(orig_trivia_file, 'trivia')
lst_squad = prepare_predict_file(orig_squad_file, 'squad')

lst_total_data = lst_musique + lst_hotpotqa + lst_wikimultihopqa + lst_nq + lst_trivia + lst_squad

output_path = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd')

save_json(output_path+'/predict.json', lst_total_data)

