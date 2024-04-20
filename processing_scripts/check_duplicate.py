import jsonlines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_name", type=str, help="dataset name.", choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad')
)
args = parser.parse_args()

lst_dev = []
with jsonlines.open('./processed_data/{}/dev_500_subsampled.jsonl'.format(args.dataset_name)) as read_file:
    for line in read_file.iter():
    	#print(line)
        lst_dev.append(line['question_id'])


lst_test = []
with jsonlines.open('./processed_data/{}/test_subsampled.jsonl'.format(args.dataset_name)) as read_file:
    for line in read_file.iter():
    	#print(line)
        if line['question_id'] in lst_dev:
            print('duplicate')
            import pdb; pdb.set_trace()
        lst_test.append(line['question_id'])
        

