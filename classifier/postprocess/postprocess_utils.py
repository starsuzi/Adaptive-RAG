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


def save_prediction_with_classified_label(total_qid_to_classification_pred, dataset_name, stepNum_result_file, dataName_to_multi_one_zero_file, output_path):
    qid_to_classification_pred = {}
    qid_to_classification_pred_option = {}
    total_stepNum = 0

    for qid in total_qid_to_classification_pred.keys():
        
        if dataset_name != total_qid_to_classification_pred[qid]['dataset_name']:
            continue

        predicted_option = total_qid_to_classification_pred[qid]['prediction']
        
        if predicted_option == 'C':
            total_dict_qid_to_stepNum = load_json(stepNum_result_file)
            stepNum = total_dict_qid_to_stepNum[qid]

        elif predicted_option == 'B':
            stepNum = 1

        elif predicted_option == 'A':
            stepNum = 0

        pred = load_json(dataName_to_multi_one_zero_file[dataset_name][predicted_option])[qid]
        qid_to_classification_pred[qid] = pred
        qid_to_classification_pred_option[qid] = {'prediction' : pred, 'option' : predicted_option, 'stepNum' : stepNum}
        total_stepNum = total_stepNum + stepNum
    
    print('==============')
    save_json(os.path.join(output_path, dataset_name , dataset_name+'.json'), qid_to_classification_pred)
    save_json(os.path.join(output_path, dataset_name, dataset_name+'_option.json'), qid_to_classification_pred_option)
    print('StepNum')
    print(dataset_name + ': ' +str(total_stepNum))