import os

from lib import read_jsonl, write_jsonl


def main():

    set_names = ["train", "dev"]
    input_directory = os.path.join("raw_data", "musique")
    output_directory = os.path.join("processed_data", "musique")
    os.makedirs(output_directory, exist_ok=True)

    for set_name in set_names:
        processed_instances = []

        input_filepath = os.path.join(input_directory, f"musique_ans_v1.0_{set_name}.jsonl")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        raw_instances = read_jsonl(input_filepath)

        for raw_instance in raw_instances:

            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }

            number_to_answer = {}
            sentences = []
            for index, reasoning_step in enumerate(raw_instance["question_decomposition"]):
                number = index + 1
                question = reasoning_step["question"]
                for mentioned_number in range(1, 10):
                    if f"#{mentioned_number}" in reasoning_step["question"]:
                        if mentioned_number not in number_to_answer:
                            print("WARNING: mentioned_number not present in number_to_answer.")
                        else:
                            question = question.replace(f"#{mentioned_number}", number_to_answer[mentioned_number])
                answer = reasoning_step["answer"]
                number_to_answer[number] = answer
                sentence = " >>>> ".join([question.strip(), answer.strip()])
                sentences.append(sentence)

            processed_instance = {
                "question_id": raw_instance["id"],
                "question_text": raw_instance["question"],
                "contexts": [
                    {
                        "idx": index,
                        "paragraph_text": paragraph["paragraph_text"].strip(),
                        "title": paragraph["title"].strip(),
                        "is_supporting": paragraph["is_supporting"],
                    }
                    for index, paragraph in enumerate(raw_instance["paragraphs"])
                ],
                "answers_objects": [answers_object],
                "reasoning_steps": sentences,
            }
            processed_instances.append(processed_instance)

        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
