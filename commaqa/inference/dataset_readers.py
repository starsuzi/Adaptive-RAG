import os
import json
import copy
from collections import Counter, defaultdict
import requests
import re

from diskcache import Cache
from tqdm import tqdm
import ftfy
import hashlib


def get_pid_for_title_paragraph_text(title: str, paragraph_text: str) -> str:
    title = ftfy.fix_text(title.strip())
    paragraph_text = ftfy.fix_text(paragraph_text.strip())

    if paragraph_text.startswith("Wikipedia Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Wikipedia Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + " \n", "").strip()

    if paragraph_text.startswith("Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Title: " + title + " \n", "").strip()

    return title + ' ' + paragraph_text
    title = "".join([i if ord(i) < 128 else " " for i in title]).lower()
    paragraph_text = "".join([i if ord(i) < 128 else " " for i in paragraph_text]).lower()
    # return title + ' ' + paragraph_text
    title = re.sub(r" +", " ", title)
    paragraph_text = re.sub(r" +", " ", paragraph_text)

    # NOTE: This is more robust, but was done after V2 big exploration.
    # So uncomment it for rerunning evals for those experiments.
    title = re.sub(r" +", "", title)
    paragraph_text = re.sub(r" +", "", paragraph_text)

    pid = "___".join(
        [
            "pid",
            hashlib.md5(title.encode("utf-8")).hexdigest(),
            hashlib.md5(paragraph_text.encode("utf-8")).hexdigest(),
        ]
    )

    return pid


class DatasetReader:
    def __init__(self, add_paras=False, add_gold_paras=False):
        self.add_paras = add_paras
        self.add_gold_paras = add_gold_paras

    def read_examples(self, file):
        return NotImplementedError("read_examples not implemented by " + self.__class__.__name__)


def format_drop_answer(answer_json):
    if answer_json["number"]:
        return answer_json["number"]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]


cache = Cache(os.path.expanduser("~/.cache/title_queries"))


def get_title_and_para(retriever_host, retriever_port, query_title):
    @cache.memoize()
    def _get_title_and_para(query_title):
        params = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": query_title,
            "max_hits_count": 1,
            "document_type": "title",
        }
        url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
        result = requests.post(url, json=params)
        if result.ok:
            result = result.json()
            retrieval = result["retrieval"]
            if not retrieval:
                return None
            if query_title != retrieval[0]["title"]:
                print(f"WARNING: {query_title} != {retrieval[0]['title']}")
            return (retrieval[0]["title"], retrieval[0]["paragraph_text"])
        else:
            return None

    return _get_title_and_para(query_title)


class MultiParaRCReader(DatasetReader):
    def __init__(
        self,
        add_paras=False,
        add_gold_paras=False,
        add_pinned_paras=False,
        pin_position="no_op",
        remove_pinned_para_titles=False,
        max_num_words_per_para=None,
        retriever_host=None,
        retriever_port=None,
    ):
        super().__init__(add_paras, add_gold_paras)
        self.add_pinned_paras = add_pinned_paras
        self.pin_position = pin_position
        self.remove_pinned_para_titles = remove_pinned_para_titles
        self.max_num_words_per_para = max_num_words_per_para
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port

        self.qid_to_external_paras = defaultdict(list)
        self.qid_to_external_titles = defaultdict(list)

    def read_examples(self, file):

        with open(file, "r") as input_fp:

            for line in tqdm(input_fp):

                if not line.strip():
                    continue

                input_instance = json.loads(line)

                qid = input_instance["question_id"]
                query = question = input_instance["question_text"]
                answers_objects = input_instance["answers_objects"]

                formatted_answers = [  # List of potentially validated answers. Usually it's a list of one item.
                    tuple(format_drop_answer(answers_object)) for answers_object in answers_objects
                ]

                answer = Counter(formatted_answers).most_common()[0][0]

                output_instance = {
                    "qid": qid,
                    "query": query,
                    "answer": answer,
                    "question": question,
                }

                for paragraph in input_instance.get("pinned_contexts", []) + input_instance["contexts"]:
                    assert not paragraph["paragraph_text"].strip().startswith("Title: ")
                    assert not paragraph["paragraph_text"].strip().startswith("Wikipedia Title: ")

                title_paragraph_tuples = []

                if self.add_pinned_paras:
                    for paragraph in input_instance["pinned_contexts"]:
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        if (title, paragraph_text) not in title_paragraph_tuples:
                            title_paragraph_tuples.append((title, paragraph_text))

                if self.add_paras:
                    assert not self.add_gold_paras, "enable only one of the two: add_paras and add_gold_paras."
                    for paragraph in input_instance["contexts"]:
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        if (title, paragraph_text) not in title_paragraph_tuples:
                            title_paragraph_tuples.append((title, paragraph_text))

                if self.add_gold_paras:
                    assert not self.add_paras, "enable only one of the two: add_paras and add_gold_paras."
                    for paragraph in input_instance["contexts"]:
                        if not paragraph["is_supporting"]:
                            continue
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        if (title, paragraph_text) not in title_paragraph_tuples:
                            title_paragraph_tuples.append((title, paragraph_text))

                if self.max_num_words_per_para is not None:
                    title_paragraph_tuples = [
                        (title, " ".join(paragraph_text.split(" ")[: self.max_num_words_per_para]))
                        for title, paragraph_text in title_paragraph_tuples
                    ]

                output_instance["titles"] = [e[0] for e in title_paragraph_tuples]
                output_instance["paras"] = [e[1] for e in title_paragraph_tuples]

                if self.remove_pinned_para_titles and "pinned_contexts" in input_instance:
                    for paragraph in input_instance["pinned_contexts"]:
                        while paragraph["title"] in output_instance["titles"]:
                            index = output_instance["titles"].index(paragraph["title"])
                            output_instance["titles"].pop(index)
                            output_instance["paras"].pop(index)

                pids = [
                    get_pid_for_title_paragraph_text(title, paragraph_text)
                    for title, paragraph_text in zip(output_instance["titles"], output_instance["paras"])
                ]
                output_instance["pids"] = pids

                output_instance["real_pids"] = [
                    paragraph["id"]
                    for paragraph in input_instance["contexts"]
                    if paragraph["is_supporting"] and "id" in paragraph
                ]

                for para in output_instance["paras"]:
                    assert not para.strip().startswith("Title: ")
                    assert not para.strip().startswith("Wikipedia Title: ")

                # Backup Paras and Titles are set so that we can filter from the original set
                # of paragraphs again and again.
                if "paras" in output_instance:
                    output_instance["backup_paras"] = copy.deepcopy(output_instance["paras"])
                    output_instance["backup_titles"] = copy.deepcopy(output_instance["titles"])

                if "valid_titles" in input_instance:
                    output_instance["valid_titles"] = input_instance["valid_titles"]

                output_instance["metadata"] = {}
                output_instance["metadata"]["level"] = input_instance.get("level", None)
                output_instance["metadata"]["type"] = input_instance.get("type", None)
                output_instance["metadata"]["answer_type"] = input_instance.get("answer_type", None)
                output_instance["metadata"]["simplified_answer_type"] = input_instance.get(
                    "simplified_answer_type", None
                )

                if self.add_pinned_paras:
                    assert len(input_instance["pinned_contexts"]) == 1
                    output_instance["metadata"]["pinned_para"] = input_instance["pinned_contexts"][0]["paragraph_text"]
                    output_instance["metadata"]["pinned_title"] = input_instance["pinned_contexts"][0]["title"]
                output_instance["metadata"]["pin_position"] = self.pin_position

                output_instance["metadata"]["gold_titles"] = []
                output_instance["metadata"]["gold_paras"] = []
                output_instance["metadata"]["gold_ids"] = []
                for paragraph in input_instance["contexts"]:
                    if not paragraph["is_supporting"]:
                        continue
                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    output_instance["metadata"]["gold_titles"].append(title)
                    output_instance["metadata"]["gold_paras"].append(paragraph_text)
                    output_instance["metadata"]["gold_ids"].append(paragraph.get("id", None))

                yield output_instance
