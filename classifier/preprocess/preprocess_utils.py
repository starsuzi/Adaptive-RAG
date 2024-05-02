import json, jsonlines
import os, sys

def load_json(json_file_path):
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def save_json(json_file_path, json_data):
    if not os.path.exists(os.path.dirname(json_file_path)): 
        os.makedirs(os.path.dirname(json_file_path)) 
    
    with open(json_file_path, "w") as output_file:
        json.dump(json_data, output_file, indent=4, sort_keys=True)
        
    print(json_file_path)

def get_overlapped_qid():
    lst_overlapped_question = []
    with open(os.path.join("raw_data", "musique", 'dev_test_singlehop_questions_v1.0.json')) as input_file:
        single_json_data = json.load(input_file)
    for id_question in single_json_data['natural_questions']:
        lst_overlapped_question.append(id_question['question'])
    return lst_overlapped_question


def get_binary_min_len(lst_dataset_name):
    min_len = float("inf")
    for dataset_name in lst_dataset_name:
        input_file = os.path.join("classifier", "data", 'musique_hotpot_wiki2_nq_tqa_sqd', 'binary', f'{dataset_name}_valid.json')
        json_data = load_json(input_file)
        min_len = len(json_data) if len(json_data) < min_len else min_len
    return min_len

def concat_and_save_binary_silver(binary_input_file, silver_input_file, output_file, min_len = sys.maxsize):
    json_data_binary = load_json(binary_input_file)
    json_data_silver = load_json(silver_input_file)

    lst_silver_ids = [i['id'] for i in json_data_silver]
    json_data_binary = [i for i in json_data_binary if i['id'] not in lst_silver_ids]

    lst_total = json_data_binary + json_data_silver[:min_len]

    save_json(output_file, lst_total)

    print(len(lst_total))

def save_inductive_bias_musique(input_file, output_file):
    lst_dict_final = []

    with jsonlines.open(input_file, 'r') as input_file:
        for line in input_file:
            dict_question_complexity = {}
            
            dict_question_complexity['id'] = line['id']
            dict_question_complexity['question'] = line['question']
            dict_question_complexity['answer_description'] = 'multi'
            dict_question_complexity['answer'] = 'C'
            dict_question_complexity['dataset_name'] = 'musique'

            lst_dict_final.append(dict_question_complexity)
                    
    save_json(output_file, lst_dict_final)

def save_inductive_bias_single_data(input_file, output_file, dataset_name, set_name):
    json_data = load_json(input_file)

    lst_dict_final = []

    if set_name == 'train' and dataset_name == 'nq':
        lst_overlapped_question = get_overlapped_qid()

    for idx, data in enumerate(json_data):
        if set_name == 'train' and dataset_name == 'nq':
            if data['question'] in lst_overlapped_question:
                    continue

        dict_question_complexity = {}

        dict_question_complexity['id'] = 'single_' + dataset_name + f'_{set_name}_'+str(idx)
        dict_question_complexity['question'] = data['question']
        dict_question_complexity['answer_description'] = 'single'
        dict_question_complexity['answer'] = 'B'
        dict_question_complexity['dataset_name'] = dataset_name

        lst_dict_final.append(dict_question_complexity)

    save_json(output_file, lst_dict_final)

def save_inductive_bias_hotpotqa(input_file, output_file):
    json_data = load_json(input_file)

    lst_dict_final = []

    for idx, data in enumerate(json_data):
        dict_question_complexity = {}

        dict_question_complexity['id'] = data['_id']
        dict_question_complexity['question'] = data['question']
        dict_question_complexity['answer_description'] = 'multi'
        dict_question_complexity['answer'] = 'C'
        dict_question_complexity['dataset_name'] = 'hotpotqa'

        lst_dict_final.append(dict_question_complexity)

    save_json(output_file, lst_dict_final)



def save_inductive_bias_2wikimultihopqa(input_file, output_file):
    json_data = load_json(input_file)

    lst_dict_final = []

    for idx, data in enumerate(json_data):
        dict_question_complexity = {}

        dict_question_complexity['id'] = data['_id']
        dict_question_complexity['question'] = data['question']
        dict_question_complexity['answer_description'] = 'multi'
        dict_question_complexity['answer'] = 'C'
        dict_question_complexity['dataset_name'] = '2wikimultihopqa'

        lst_dict_final.append(dict_question_complexity)

    save_json(output_file, lst_dict_final)

def label_complexity(orig_file_path, zero_file_path, one_file_path, multi_file_path, dataset_name):
    lst_dict_final = []
    with jsonlines.open(orig_file_path, 'r') as input_file:
        for line in input_file:
            dict_question_complexity = {}
            dict_question_complexity['id'] = line['question_id']
            dict_question_complexity['question'] = line['question_text']

            dict_zero = load_json(zero_file_path)
            dict_one = load_json(one_file_path)
            dict_multi = load_json(multi_file_path)

            lst_multi_qid = [i for i in dict_multi.keys()]
            lst_one_qid = [i for i in dict_one.keys()]
            lst_zero_qid = [i for i in dict_zero.keys()]

            if line['question_id'] not in lst_multi_qid + lst_one_qid + lst_zero_qid:
                continue

            dict_question_complexity['dataset_name'] = dataset_name

            lst_total_answer = []

            if line['question_id'] in lst_multi_qid:
                dict_question_complexity['answer'] = 'C' #'multiple'
                lst_total_answer.append('multiple')
            if line['question_id'] in lst_one_qid:
                dict_question_complexity['answer'] = 'B' # 'one'
                lst_total_answer.append('one')
            if line['question_id'] in lst_zero_qid:
                dict_question_complexity['answer'] = 'A' #'zero'
                lst_total_answer.append('zero')
            
            dict_question_complexity['total_answer'] = lst_total_answer

            lst_dict_final.append(dict_question_complexity)

    return lst_dict_final

def prepare_predict_file(orig_file_path, dataset_name):
    lst_dict_final = []
    with jsonlines.open(orig_file_path, 'r') as input_file:
        for line in input_file:
            dict_question_doc_count = {}
            dict_question_doc_count['id'] = line['question_id']
            dict_question_doc_count['question'] = line['question_text']

            dict_question_doc_count['dataset_name'] = dataset_name

            lst_total_answer = []
            dict_question_doc_count['answer'] = ''

            dict_question_doc_count['total_answer'] = lst_total_answer

            lst_dict_final.append(dict_question_doc_count)

    return lst_dict_final


def count_stepNum(pred_file):
    dict_qid_to_stepNum = {}
    total_stepNum = 0
    stepNum = 0
    new_qid_flag = False

    with open(pred_file, "r") as f:
        for line in f:
            if line == '\n':
                new_qid_flag = True
                if 'qid' in locals():
                    dict_qid_to_stepNum[qid] = stepNum + 1
                    total_stepNum = total_stepNum + stepNum + 1
                stepNum = 0
                continue

            if new_qid_flag:
                qid = line.strip()
                new_qid_flag = False

            if 'Exit? No.' in line:
                stepNum = stepNum + 1
    
    # last qid
    dict_qid_to_stepNum[qid] = stepNum + 1
    total_stepNum = total_stepNum + stepNum + 1

    output_file = '/'.join(pred_file.split('/')[:-1]) + '/stepNum.json'
    save_json(output_file, dict_qid_to_stepNum)

    print(total_stepNum)

    return dict_qid_to_stepNum