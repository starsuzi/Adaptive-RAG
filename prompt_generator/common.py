from typing import List, Dict, Tuple, Union, Any
from functools import lru_cache
import json
import random
import copy
import re


random.seed(13370)  # Don't change.


def safe_sample(items: List[Any], count: int) -> List[Any]:
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


@lru_cache(maxsize=None)
def get_spacy_object():
    import spacy

    return spacy.load("en_core_web_sm")


def clip_paragraph_text(paragraph_text: str, max_tokens: int = 250) -> str:
    spacy_object = get_spacy_object()
    paragraph_object = spacy_object(paragraph_text)
    paragraph_sents = paragraph_object.sents

    clipped_paragraph_tokens = 0
    clipped_paragraph_text = ""
    for sent in paragraph_sents:
        if clipped_paragraph_tokens + len(sent) >= max_tokens:
            break
        clipped_paragraph_text += sent.text_with_ws
        clipped_paragraph_tokens += len(sent)
    return clipped_paragraph_text


def clip_paragraphs(paragraphs: List[Dict], max_tokens: int = 250):
    paragraphs = copy.deepcopy(paragraphs)
    for paragraph in paragraphs:

        if paragraph["is_supporting"] or paragraph["is_pinned"]:
            continue

        paragraph_text = paragraph["paragraph_text"]
        clipped_paragraph_text = clip_paragraph_text(paragraph_text, max_tokens)
        paragraph["paragraph_text"] = clipped_paragraph_text

    return paragraphs


class PromptGenerator:
    def __init__(
        self,
        input_file_path: str,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        self._instances = read_jsonl(input_file_path)
        self._demonstration_delimiter = demonstration_delimiter
        self._one_demonstration_per_instance = one_demonstration_per_instance

    def _generate(self, instance: Dict) -> str:
        raise NotImplementedError

    def generate(self) -> str:
        def instance_to_header(instance):
            return "# METADATA: " + json.dumps({"qid": instance["question_id"]})

        all_demonstrations_with_headers = []
        for instance in self._instances:
            local_demonstrations_with_headers = [
                "\n".join([instance_to_header(instance), demonstration]).strip()
                for demonstration in self._generate(instance)
            ]
            if len(local_demonstrations_with_headers) > 1 and self._one_demonstration_per_instance:
                all_demonstrations_with_headers.append(random.choice(local_demonstrations_with_headers))
            else:
                all_demonstrations_with_headers += local_demonstrations_with_headers

        generated_output = self._demonstration_delimiter.join(all_demonstrations_with_headers)
        return generated_output


class QAPromptGenerator(PromptGenerator):
    def __init__(
        self,
        input_file_path: str,
        qa_type: str,
        context_type: str,
        model_name: str,
        distractor_count: Union[int, Tuple[int, int]] = 0,
        max_paragraph_tokens: int = 250,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
        pinned_at_bottom: bool = True,
    ):
        super().__init__(input_file_path, demonstration_delimiter, one_demonstration_per_instance)
        assert qa_type in ("direct", "cot"), f"qa_type must be in direct, cot. Found {qa_type}."
        self._qa_type = qa_type
        assert context_type in (
            "no",
            "gold_with_distractors",
        ), f"context_type must be in no or gold_with_distractors. Found {context_type}."
        self._context_type = context_type
        if context_type == "no":
            assert distractor_count == 0
        assert isinstance(distractor_count, int) or isinstance(distractor_count, tuple)
        self._distractor_count = distractor_count
        self._max_paragraph_tokens = max_paragraph_tokens
        assert model_name in ("codex", "flan_t5"), f"model_name must be in codex, flan_t5. Found {model_name}."
        self._model_name = model_name
        assert pinned_at_bottom in (True, False), "pinned_at_bottom must be True or False."
        self._pinned_at_bottom = pinned_at_bottom

    def _generate(self, instance: Dict) -> List[Dict]:

        cot_sents = []
        reasoning_steps = instance["reasoning_steps"]

        # Gather Gold Paragraphs and Gold CoT
        gold_paragraphs = instance.get("pinned_contexts", [])
        taken_gold_title_paras = []
        for paragraph in gold_paragraphs:
            paragraph["is_pinned"] = True
            paragraph["is_supporting"] = True
            title_para = (paragraph["title"], paragraph["paragraph_text"])
            taken_gold_title_paras.append(title_para)

        for reasoning_step in reasoning_steps:

            if self._qa_type == "cot":
                cot_sents.append(reasoning_step["cot_sent"])

            assert len(reasoning_step["paragraphs"]) == 1
            paragraph = reasoning_step["paragraphs"][0]

            if not paragraph["title"]:
                continue

            title_para = (paragraph["title"], paragraph["paragraph_text"])

            if title_para in taken_gold_title_paras:
                continue

            paragraph["is_supporting"] = True
            paragraph["is_pinned"] = False

            gold_paragraphs.append(paragraph)
            taken_gold_title_paras.append(title_para)

        if self._context_type == "no":
            gold_paragraphs = [paragraph for paragraph in gold_paragraphs if paragraph["is_pinned"]]

        # Gather Distractor Paragraphs
        distractor_paragraphs = []

        if self._context_type == "gold_with_distractors":

            if isinstance(self._distractor_count, int):
                distractor_count = self._distractor_count
            else:
                distractor_count = random.randint(*self._distractor_count)
            candidate_distractor_paragraphs = [
                paragraph for paragraph in instance["contexts"] if not paragraph["is_supporting"]
            ]
            candidate_distractor_paragraphs = [
                paragraph
                for paragraph in candidate_distractor_paragraphs
                if (paragraph["title"], paragraph["paragraph_text"]) not in taken_gold_title_paras
            ]
            for paragraph in candidate_distractor_paragraphs:
                assert not paragraph.get("is_supporting", False)
                paragraph["is_supporting"] = False
                paragraph["is_pinned"] = False
            distractor_paragraphs = safe_sample(candidate_distractor_paragraphs, distractor_count)

        # Put all paragraphs together in context
        all_paragraphs = gold_paragraphs + distractor_paragraphs
        all_paragraphs = clip_paragraphs(all_paragraphs, max_tokens=self._max_paragraph_tokens)

        random.shuffle(all_paragraphs)
        all_paragraphs = sorted(all_paragraphs, key=lambda e: int(e["is_pinned"]), reverse=not self._pinned_at_bottom)

        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in all_paragraphs
            ]
        ).strip()

        # Prepare Answer
        assert len(instance["answers_objects"]) == 1
        assert len(instance["answers_objects"][0]["spans"]) == 1
        answer_text = instance["answers_objects"][0]["spans"][0]

        # Prepare CoT
        cot_text = re.sub(r" +", " ", " ".join(cot_sents))

        # Prepare Full Demonstration
        if self._qa_type == "direct":
            output_text = "A: " + answer_text
            qn_pretext = "Q:"
        elif self._qa_type == "cot":
            output_text = "A: " + cot_text
            qn_pretext = "Q:"
        else:
            raise Exception(f"Encountered unknown choice of qa_type {self._qa_type}.")

        question_instruction = ""
        if self._model_name == "flan" and self._qa_type == "direct":
            question_instruction = "Answer the following question.\n"
        elif self._model_name == "flan" and "cot" in self._qa_type:
            question_instruction = "Answer the following question by reasoning step-by-step.\n"

        question_text = instance["question_text"]
        demonstration = "\n".join(
            [context_text, "", f"{qn_pretext} {question_instruction}{question_text}", output_text]
        ).strip()

        return [demonstration]


class NoContextOpenRetrieverPromptGenerator(PromptGenerator):
    def __init__(
        self,
        input_file_path: str,
        model_name: str,
        demonstration_delimiter: str = "\n\n\n",
        one_demonstration_per_instance: bool = False,
    ):
        super().__init__(input_file_path, demonstration_delimiter, one_demonstration_per_instance)
        assert model_name in ("codex", "flan_t5"), f"model_name must be in codex, flan_t5. Found {model_name}."
        self._model_name = model_name

    def _generate(self, instance: Dict) -> List[Dict]:
        question_text = instance["question_text"]

        # NOTE: pinned_contexts are added by default. Can make it configurable later if needed.
        pinned_paragraphs = instance.get("pinned_contexts", [])
        context_text = "\n\n".join(
            [
                "Wikipedia Title: "
                + paragraph["title"]
                + "\n"
                + paragraph["paragraph_text"].strip().replace("\n", " ").strip()
                for paragraph in pinned_paragraphs
            ]
        ).strip()

        # NOTE: Make sure to retain the order of the paragraph titles.
        # Since they follow the reasoning steps, the order might help learn the task better.
        gold_titles = []
        avoid_gold_titles = set(paragraph["title"] for paragraph in instance.get("pinned_contexts", []))
        for reasoning_step in instance["reasoning_steps"]:
            assert len(reasoning_step["paragraphs"]) == 1
            title = reasoning_step["paragraphs"][0]["title"]
            if not title:
                continue
            if title in gold_titles or title in avoid_gold_titles:
                continue
            gold_titles.append(reasoning_step["paragraphs"][0]["title"])

        retrieval_count = len(gold_titles)

        question_prefix = ""
        if self._model_name == "flan":
            question_prefix = "Answer the following question.\n"

        input_text = (
            f"{question_prefix}The question is: '{question_text}'. "
            f"Generate titles of {retrieval_count} Wikipedia pages that have relevant information to answer this question."
        )

        if self._model_name == "flan":
            output_text = ", ".join(gold_titles).strip()
        else:
            output_text = "[" + ", ".join(['"' + e.replace('"', "'") + '"' for e in gold_titles]) + "]"

        demonstration = "\n".join([context_text, "", f"Q: {input_text}", f"A: {output_text}"]).strip()

        return [demonstration]
