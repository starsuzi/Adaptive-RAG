import json
import sys

import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


examples = []
with open(sys.argv[1], "r") as input_fp:
    for line in input_fp:
        input_json = json.loads(line.strip())
        question = input_json["question"]
        rationale = input_json["answer"]
        answer = rationale.split("####")[-1].strip()
        try:
            answer = locale.atoi(answer)
        except ValueError:
            try:
                answer = locale.atof(answer)
            except ValueError:
                print("Can not parse: " + answer)
        examples.append((question, answer, rationale))

qa_pairs = []
for eg_idx, eg in enumerate(examples):
    drop_answer = {"number": eg[1], "date": {"day": "", "month": "", "year": ""}, "spans": []}
    qa_pairs.append({"question": eg[0], "answer": drop_answer, "query_id": str(eg_idx), "rationale": eg[2]})

    output_json = {"1": {"passage": "", "qa_pairs": qa_pairs}}
with open(sys.argv[2], "w") as output_fp:
    output_fp.write(json.dumps(output_json, indent=2))
