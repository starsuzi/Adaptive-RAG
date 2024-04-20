import os

from lib import read_json, write_jsonl


def main():

    set_names = ["train", "dev"]

    input_directory = os.path.join("raw_data", "2wikimultihopqa")
    output_directory = os.path.join("processed_data", "2wikimultihopqa")
    os.makedirs(output_directory, exist_ok=True)

    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        input_filepath = os.path.join(input_directory, f"{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        raw_instances = read_json(input_filepath)

        for raw_instance in raw_instances:

            question_id = raw_instance["_id"]
            question_text = raw_instance["question"]
            raw_contexts = raw_instance["context"]

            supporting_titles = list(set([e[0] for e in raw_instance["supporting_facts"]]))

            evidences = raw_instance["evidences"]
            reasoning_steps = [" ".join(evidence) for evidence in evidences]

            processed_contexts = []
            for index, raw_context in enumerate(raw_contexts):
                title = raw_context[0]
                paragraph_text = " ".join(raw_context[1]).strip()
                is_supporting = title in supporting_titles
                processed_contexts.append(
                    {
                        "idx": index,
                        "title": title.strip(),
                        "paragraph_text": paragraph_text,
                        "is_supporting": is_supporting,
                    }
                )

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            answers_objects = [answers_object]

            processed_instance = {
                "question_id": question_id,
                "question_text": question_text,
                "answers_objects": answers_objects,
                "contexts": processed_contexts,
                "reasoning_steps": reasoning_steps,
            }

            processed_instances.append(processed_instance)

        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
