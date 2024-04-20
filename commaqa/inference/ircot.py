import re
import copy
import json
import time
from typing import List
from functools import lru_cache
import random

import requests
from rapidfuzz import fuzz

from commaqa.inference.prompt_reader import read_prompt
from commaqa.inference.data_instances import QuestionAnsweringStep, QuestionGenerationStep, Task
from commaqa.inference.model_search import ParticipantModel
from commaqa.models.gpt3generator import GPT3Generator
from commaqa.models.llm_client_generator import LLMClientGenerator
from commaqa.inference.dataset_readers import get_pid_for_title_paragraph_text


random.seed(100)  # Don't change.


@lru_cache(maxsize=None)
def get_spacy_object():
    import spacy

    return spacy.load("en_core_web_sm")


def is_reasoning_sentence(sentence: str) -> bool:
    starters = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
    for starter in starters:
        if sentence.lower().startswith(starter):
            return True

    regex = re.compile("(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
    match = bool(re.match(regex, sentence))
    if match:
        return True

    return False


def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]


def safe_post_request(url, params):
    for _ in range(10):
        try:
            return requests.post(url, json=params)
        except:
            print("Post request didn't succeed. Will wait 20s and retry.")
            time.sleep(20)
    raise Exception("Post request couldn't succeed after several attempts.")


def remove_wh_words(text: str) -> str:
    wh_words = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
    words = [word for word in text.split(" ") if word.strip().lower() not in wh_words]
    text = " ".join(words)
    return text


def get_real_pid_for_title_paragraph_text(
    source_corpus_name: str, retriever_host: str, retriever_port: str, title, paragraph_text
) -> str:
    query_text = " ".join(paragraph_text.split(" ")[:30])
    params = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "allowed_titles": [title],
        "query_text": query_text,
        "max_hits_count": 20,
        "corpus_name": source_corpus_name,
        "document_type": "paragraph_text",
    }

    url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
    result = safe_post_request(url, params)

    result = result.json()
    retrieval = result["retrieval"]

    if not retrieval:
        print("WARNING: Not para with the same title retrieved.")
        return ""

    def para_similarity_func(retrieval_):
        return (
            float(retrieval_["title"].lower() == title.lower())
            + get_token_similarity(retrieval_["paragraph_text"], paragraph_text) / 100
        )

    retrieval = sorted(retrieval, key=para_similarity_func, reverse=True)[0]

    retrieved_title = retrieval["title"]
    retrieved_para = retrieval.get("paragraph_text", "")  # backoff for natcq
    retrieved_id = retrieval["id"]  # has to be there.
    assert retrieved_id

    if retrieved_title != title:
        print("WARNING: Para with the same title couldn't be identified.")
        retrieved_id = ""
    if retrieved_para != paragraph_text:
        print("WARNING: Para with the same paragraph_text couldn't be identified.")
        retrieved_id = ""

    return retrieved_id


def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:

    if new_title in existing_titles and new_para in existing_paras:
        return True

    assert match_threshold > 1.0, "The threshold is 0-100 scaled."

    assert len(existing_titles) == len(existing_paras)
    for existing_title, existing_para in zip(existing_titles, existing_paras):
        condition_1 = fuzz.ratio(existing_title, new_title) >= match_threshold
        condition_2 = fuzz.ratio(existing_para, new_para) >= match_threshold
        if condition_1 and condition_2:
            return True
    return False


def para_to_text(title: str, para: str, max_num_words: int) -> int:
    # Note: the split and join must happen before the attaching title+para.
    # also don't split() because that disrupts the new lines.
    para = " ".join(para.split(" ")[:max_num_words])
    para = (
        para.strip()
        if para.strip().startswith("Wikipedia Title: ")
        else "Wikipedia Title: " + title + "\n" + para.strip()
    )
    return para


def assert_unique_titles_paras(titles: List[str], paras: List[str]) -> bool:
    titles_paras = [(title, para) for title, para in zip(titles, paras)]
    assert len(titles_paras) == len(set(titles_paras))


def get_token_similarity(str_1: str, str_2: str) -> float:
    return fuzz.token_sort_ratio(str_1.lower(), str_2.lower())


def add_and_reorder_if_pinned(titles, paras, pinned_title, pinned_para, pin_position):

    if pinned_title is not None or pinned_para is not None:
        assert pinned_title is not None and pinned_para is not None

        if pinned_para not in paras:
            titles.insert(0, pinned_title)
            paras.insert(0, pinned_para)

        pin_index = paras.index(pinned_para)
        assert titles[pin_index].lower().strip() == pinned_title.lower().strip()

        if pin_position == "no_op":
            return titles, paras

        elif pin_position == "top":
            titles.pop(pin_index)
            paras.pop(pin_index)
            titles.insert(0, pinned_title)
            paras.insert(0, pinned_para)

        elif pin_position == "bottom":
            titles.pop(pin_index)
            paras.pop(pin_index)
            titles.append(pinned_title)
            paras.append(pinned_para)

        else:
            raise Exception(f"Unknown pin_position {pin_position}")

    return titles, paras


class AnswerExtractor(ParticipantModel):
    def __init__(
        self,
        regex,
        next_model="[EOQ]",
        match_all_on_failure=False,
        query_source="last_question",
        remove_last_fullstop=False,
    ):
        self.regex = re.compile(regex)
        self.next_model = next_model
        self.num_calls = 0
        self.match_all_on_failure = match_all_on_failure
        self.query_source = query_source
        self.remove_last_fullstop = remove_last_fullstop
        assert query_source in (
            "last_question",
            "last_answer",
        ), f"query_source must be either last_question or last_answer. Found {query_source}."

    def return_model_calls(self):
        return {"extract": self.num_calls}

    def query(self, state, debug=False):
        self.num_calls += 1

        new_state = state.copy()

        if self.query_source == "last_answer":
            query = new_state.data.get_last_answer()
        else:
            query = new_state.data.get_last_question()

        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]

        m = self.regex.match(query)
        if self.match_all_on_failure and not self.regex.match(query):
            m = re.compile(r"(.*)").match(query)

        if m:
            answer = m.group(1)

            if self.remove_last_fullstop and answer.endswith("."):
                answer = answer[:-1]

            if debug:
                print("EXT: " + answer)

            try:  # Hacky. Fix later. This is to handle '[\\"1,450 miles\\"]' to '["1,450 miles"]'
                json.loads(answer)
            except:
                try:
                    answer = json.dumps(json.loads(answer.encode("utf-8").decode("unicode_escape")))
                except:
                    pass

            new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
            new_state.last_output = answer
            new_state.next = self.next_model
            return new_state
        else:
            print("Answer Extractor did not find a match for input regex in {}".format(query))
            return []


class RetrieveAndResetParagraphsParticipant(ParticipantModel):
    def __init__(
        self,
        retrieval_type,
        retriever_host=None,
        retriever_port=None,
        retrieval_count=None,
        query_source="last_answer",
        cumulate_titles=False,
        document_type="title",
        source_corpus_name=None,
        dont_add_to_state=False,
        allowed_paragraph_types=None,
        dont_skip_long_paras=False,
        return_pids=False,
        return_paras=False,
        valid_titles_are_allowed_titles=False,
        set_result_as_valid_titles=False,
        global_max_num_paras=100,
        next_model=None,
        end_state="[EOQ]",
    ):

        assert retrieval_type in (
            "map_generated_to_valid_titles",
            "bm25",
        ), f"retrieval_type {retrieval_type} not among the valid choices."

        assert query_source in (
            "original_question",
            "last_answer",
            "question_or_last_generated_sentence",
        ), f"query_source {query_source} not among the valid choices."

        assert document_type in ("title", "paragraph_text", "title_paragraph_text")

        self.valid_titles_are_allowed_titles = valid_titles_are_allowed_titles
        if valid_titles_are_allowed_titles:
            assert (
                retrieval_type == "bm25"
            ), "valid_titles_are_allowed_titles is applicable only when retrieval_type is bm25."

        if set_result_as_valid_titles:
            assert (
                retrieval_type == "map_generated_to_valid_titles"
            ), "set_result_as_valid_titles is only available for map_generated_to_valid_titles retrieval type."
        self.set_result_as_valid_titles = set_result_as_valid_titles

        self.global_max_num_paras = global_max_num_paras
        self.retrieval_type = retrieval_type
        self.next_model = next_model
        self.end_state = end_state
        self.retriever_host = retriever_host
        self.retriever_port = retriever_port
        self.retrieval_count = retrieval_count
        self.document_type = document_type
        self.query_source = query_source
        self.cumulate_titles = cumulate_titles
        self.source_corpus_name = source_corpus_name
        self.dont_add_to_state = dont_add_to_state
        self.dont_skip_long_paras = dont_skip_long_paras
        self.return_pids = return_pids
        self.return_paras = return_paras
        self.num_calls = 0

        if self.return_pids and self.return_paras:
            raise Exception("Only one of return_pids or return_paras should be true.")

        if allowed_paragraph_types:
            assert isinstance(allowed_paragraph_types, list)
            self.allowed_paragraph_types = allowed_paragraph_types
        else:
            self.allowed_paragraph_types = [None]

        if retrieval_type == "bm25":
            if self.retrieval_count is None:
                raise Exception(f"retrieval_count is needed for the retrieval_type {retrieval_type}.")
            if self.source_corpus_name is None:
                raise Exception(f"source_corpus_name is needed for the retrieval_type {retrieval_type}.")

        self.retrieval_failures_so_far = 0
        self.retrieval_failures_max = 9

    def return_model_calls(self):
        return {"paragraph_retrieve_and_reset": self.num_calls}

    def query(self, state, debug=False):

        if self.query_source == "original_question":
            input_query = state.data["question"]

        elif self.query_source == "last_answer":
            input_query = state.data.get_last_answer()

        elif self.query_source == "question_or_last_generated_sentence":
            # add question to query only if generated sentences are empty. O/w use last_generated_sentence.
            question = state.data["question"]
            generated_sentences = state.data.get("generated_sentences", [])
            generated_sentences = remove_reasoning_sentences(generated_sentences)
            last_generated_sentence_str = generated_sentences[-1].strip() if generated_sentences else ""
            input_query = last_generated_sentence_str if last_generated_sentence_str else question

        else:
            raise Exception(f"Unknown query_source: {self.query_source}.")

        if self.cumulate_titles:
            selected_titles = state.data["titles"]
            selected_paras = state.data["paras"]
        else:
            selected_titles = []
            selected_paras = []

        if self.retrieval_type == "map_generated_to_valid_titles":

            try:
                # Assuming input_query will be of form: '["title3", "title7"]''
                generated_titles = json.loads(input_query)
            except:
                generated_titles = [
                    e.strip() for e in input_query.strip().replace('"', "").replace("[", "").replace("]", "").split(",")
                ]

            for generated_title in generated_titles:

                assert self.document_type == "title"
                params = {
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "query_text": generated_title,
                    "max_hits_count": self.retrieval_count,
                    "document_type": "title",
                    "corpus_name": self.source_corpus_name,
                }
                url = self.retriever_host.rstrip("/") + ":" + str(self.retriever_port) + "/retrieve"
                result = safe_post_request(url, params)

                locally_mapped_titles = set()
                if result.ok:

                    result = result.json()
                    retrieval = result["retrieval"]

                    if not retrieval:
                        continue

                    for retrieval_ in retrieval:
                        selected_title = retrieval_["title"]
                        selected_para = retrieval_.get("paragraph_text", "")  # backoff for natcq

                        locally_mapped_titles.add(selected_title)

                        if len(selected_para.split(" ")) > 600 and not self.dont_skip_long_paras:
                            print("WARNING: Discarding a retrieved paragraph as it's excessively long.")
                            continue

                        if retrieval_["corpus_name"] != self.source_corpus_name:
                            raise Exception(
                                f"The retrieved corpus name {retrieval_['corpus_name']} "
                                f"doesn't match {self.source_corpus_name}."
                            )

                        if selected_title not in selected_titles and len(selected_paras) < self.global_max_num_paras:
                            selected_titles.append(selected_title)
                            selected_paras.append(selected_para)

                else:
                    self.retrieval_failures_so_far += 1
                    if self.retrieval_failures_so_far > self.retrieval_failures_max:
                        raise Exception(
                            f"Retrieval failure exceeded max allowed times ({self.retrieval_failures_so_far} > {self.retrieval_failures_max})"
                        )
                    print(
                        f"WARNING: Retrieval of titles did not succeed {self.retrieval_failures_so_far} times. Skipping it."
                    )

            if self.set_result_as_valid_titles:
                state.data["valid_titles"] = selected_titles

        elif self.retrieval_type == "bm25":

            retrieval_method = "retrieve_from_elasticsearch"
            input_query = remove_wh_words(input_query)

            params = {
                "retrieval_method": retrieval_method,
                "query_text": input_query,
                "max_hits_count": self.retrieval_count,
                "corpus_name": self.source_corpus_name,
            }

            for allowed_paragraph_type in self.allowed_paragraph_types:

                if allowed_paragraph_type is not None:
                    params["allowed_paragraph_types"] = [allowed_paragraph_type]

                if not input_query.strip():
                    # can happen when query is based on last cot gen
                    # but it's a reasoning (non-factual) sentence.
                    continue

                if self.retrieval_type == "bm25":
                    params["document_type"] = self.document_type

                if self.valid_titles_are_allowed_titles:
                    params["allowed_titles"] = state.data["valid_titles"]

                url = self.retriever_host.rstrip("/") + ":" + str(self.retriever_port) + "/retrieve"
                result = safe_post_request(url, params)

                if result.ok:

                    result = result.json()
                    retrieval = result["retrieval"]

                    for retrieval_item in retrieval:

                        if retrieval_item["corpus_name"] != self.source_corpus_name:
                            raise Exception(
                                f"The retrieved corpus name {retrieval_item['corpus_name']} "
                                f"doesn't match {self.source_corpus_name}."
                            )

                        if len(retrieval_item["paragraph_text"].split(" ")) > 600 and not self.dont_skip_long_paras:
                            print("WARNING: Discarding a retrieved paragraph as it's excessively long.")
                            continue

                        if is_para_closely_matching(
                            selected_titles,
                            selected_paras,
                            retrieval_item["title"],
                            retrieval_item["paragraph_text"],
                        ):
                            continue

                        if len(selected_paras) >= self.global_max_num_paras:
                            continue

                        if self.valid_titles_are_allowed_titles:
                            assert retrieval_item["title"].lower().replace(" ", "") in [
                                valid_title.lower().replace(" ", "") for valid_title in state.data["valid_titles"]
                            ]

                        selected_titles.append(retrieval_item["title"])
                        selected_paras.append(retrieval_item["paragraph_text"])

                else:
                    self.retrieval_failures_so_far += 1
                    if self.retrieval_failures_so_far > self.retrieval_failures_max:
                        raise Exception(
                            f"Retrieval failure exceeded max allowed times ({self.retrieval_failures_so_far} > {self.retrieval_failures_max})"
                        )
                    print(
                        f"WARNING: Retrieval of titles did not succeed {self.retrieval_failures_so_far} times. Skipping it."
                    )

        else:
            raise Exception(
                f"retrieval_type must be one of 'map_generated_to_valid_titles', 'bm25'. Found {self.retrieval_type}."
            )

        self.num_calls += 1

        answer = json.dumps(selected_titles)

        if self.return_pids:
            pids = [
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(selected_titles, selected_paras)
            ]
            answer = json.dumps(pids)

        if self.return_paras:
            answer = json.dumps(
                [{"title": title, "paragraph_text": para} for title, para in zip(selected_titles, selected_paras)]
            )

        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=answer, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state

        if not self.dont_add_to_state:
            new_state.data["paras"] = selected_paras
            new_state.data["titles"] = selected_titles

        return new_state


class CopyQuestionParticipant(ParticipantModel):
    """
    Generates question by copying the question field from the data json.
    """

    def __init__(
        self,
        next_model=None,
        end_state="[EOQ]",
        eoq_after_n_calls=1,
    ):
        self.next_model = next_model
        self.end_state = end_state
        self.num_calls = 0
        self.eoq_after_n_calls = eoq_after_n_calls

    def return_model_calls(self):
        return {"copy_question": self.num_calls}

    def query(self, state, debug=False):

        if (self.num_calls + 1) % (self.eoq_after_n_calls + 1) == 0:
            output = self.end_state
        else:
            output = state.data["question"].strip()

        self.num_calls += 1

        new_state = state.copy()

        new_state.data.add_qgen(QuestionGenerationStep(question=output, score=0, participant=state.next))

        if output == self.end_state:
            new_state.next = self.end_state
        else:
            new_state.data.add_task(Task(task_question=None, task_participant=new_state.next))
            new_state.next = self.next_model

        return [new_state]


class StepByStepLLMTitleGenParticipant(ParticipantModel):
    """Goes with StepByStepCOTGenParticipant"""

    def __init__(
        self,
        retrieval_count,
        prompt_file,
        prompt_reader_args,
        show_so_far_titles,
        show_so_far_paras,
        show_so_far_cot,
        prompt_question="",
        max_para_num_words=350,
        gen_model="gpt3",
        next_model=None,
        question_prefix="",
        end_state="[EOQ]",
        **kwargs,
    ) -> None:

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        assert isinstance(retrieval_count, int)
        assert isinstance(show_so_far_titles, bool)
        assert isinstance(show_so_far_paras, bool)
        assert isinstance(show_so_far_cot, bool)

        self.retrieval_count = retrieval_count
        self.show_so_far_titles = show_so_far_titles
        self.show_so_far_paras = show_so_far_paras
        self.show_so_far_cot = show_so_far_cot

        tpc_combination = "".join(
            ("Y" if show_so_far_titles else "N", "Y" if show_so_far_paras else "N", "Y" if show_so_far_cot else "N")
        )
        valid_tpc_combinations = ("NNN", "NYN", "NNY", "YNY", "YNN", "YYN", "YYY")
        # The NNN and NYN are only for the base condition, when no contextual info is available
        # NNN when paras are not pinned, NYN when they are pinned.
        assert tpc_combination in valid_tpc_combinations, f"given tpc_combination ({tpc_combination}) is not valid."

        if prompt_file:
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            print("WARNING: Using StepByStepLLMTitleGenParticipant without any prompt.")
            self.prompt = ""
        self.prompt_question = prompt_question
        self.question_prefix = question_prefix

        self.max_para_num_words = max_para_num_words
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMClientGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

    def return_model_calls(self):
        return {"step_by_step_retrieve": self.num_calls}

    def query(self, state, debug=False):

        paras_text = ""
        if self.show_so_far_paras:
            zipped_titles_paras = list(zip(state.data["titles"], state.data["paras"]))
            paragraphs = [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
            paras_text = "\n\n".join(paragraphs).strip()
            if not paragraphs:
                print(
                    "WARNING: Found a case of non-contexual question on contextual prompt. Prompt isn't 'trained' for it."
                )

        titles_text = ""
        if self.show_so_far_titles:
            so_far_titles = list(dict.fromkeys(state.data.get("titles", [])).keys())
            if so_far_titles:
                titles_text = "So far collected Wikipedia page titles: "
                titles_text += "[" + ", ".join(['"' + str(e) + '"' for e in so_far_titles]) + "]"
            else:
                print(
                    "WARNING: Found a case of non-contexual question on contextual prompt. Prompt isn't 'trained' for it."
                )
                titles_text = "So far no wikipedia page titles have been collected."

        cot_text = ""
        if self.show_so_far_cot:
            so_far_cot = " ".join(state.data.get("generated_sentences", [])).strip()
            if so_far_cot:
                cot_text = f"So far collected evidence: {so_far_cot}"
            else:
                print(
                    "WARNING: Found a case of non-contexual question on contextual prompt. Prompt isn't 'trained' for it."
                )
                cot_text = "So far no evidence has been collected."

        multihop_question = state.data["question"]
        question_text = f"Q: {self.question_prefix}The question is: '{multihop_question}'. "

        if self.prompt_question:
            question_text += self.prompt_question
        else:
            question_text += (
                f"Read the information given above to answer this question, and "
                f"generate titles of {self.retrieval_count} additional Wikipedia pages that have relevant information to answer this question."
            )

        test_example_str = "\n\n".join([paras_text, titles_text + "\n" + cot_text, question_text]).strip()
        test_example_str += "\n" + "A: "
        test_example_str = re.sub(r"\n\n+", "\n\n", test_example_str)

        prompt = "\n\n\n".join([self.prompt, test_example_str]).strip()

        output_text_scores = self.generator.generate_text_sequence(prompt)

        if len(output_text_scores) > 1:
            print("Can not handle more than one answer for this model yet" + "\n" + str(output_text_scores))

        generated_titles_str = output_text_scores[0][0].strip()

        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=generated_titles_str, score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state

        self.num_calls += 1

        return new_state


class StepByStepCOTGenParticipant(ParticipantModel):
    """
    Keeps a state of generated COT, and continues it with one sentence at a time.
    The context fed to the COT generator can be changed by changing state.data["titles"]
    """

    def __init__(
        self,
        prompt_file="",
        prompt_reader_args=None,
        add_context=True,
        answer_extractor_regex=".* answer is (.*)",
        answer_extractor_remove_last_fullstop=True,
        terminal_return_type="titles",
        generation_type="sentences",
        reset_queries_as_sentences=False,
        max_num_sentences=10,
        terminal_state_next_model=None,
        shuffle_paras=False,
        disable_exit=False,
        max_para_num_words=350,
        question_prefix="",
        gen_model="gpt3",
        next_model=None,
        end_state="[EOQ]",
        **kwargs,
    ):

        import spacy  # Kept here because it's almost always not required, and it's slow.

        self.num_calls = 0
        self.next_model = next_model
        self.end_state = end_state

        if prompt_file:
            prompt_reader_args = prompt_reader_args or {}
            prompt_reader_args["file_path"] = prompt_file
            self.prompt = read_prompt(**prompt_reader_args)
        else:
            print("WARNING: Using StepByStepCOTGenParticipant without any prompt.")
            self.prompt = ""

        self.max_para_num_words = max_para_num_words
        if gen_model == "gpt3":
            self.generator = GPT3Generator(**kwargs)
        elif gen_model == "llm_api":
            self.generator = LLMClientGenerator(**kwargs)
        else:
            raise ValueError("Unknown gen_model: " + gen_model)

        if disable_exit:
            assert terminal_return_type is None, "When disable_exit is True, terminal_return_type must be None."
        else:
            if terminal_return_type not in ("answer", "titles", "pids"):
                raise Exception(
                    f"When disable_exit is False, terminal_return_type has to be one of answer or titles."
                    f"Found {terminal_return_type}."
                )

        assert generation_type in ("sentences", "queries")

        self.add_context = add_context
        self.answer_extractor_regex = re.compile(answer_extractor_regex)
        self.answer_extractor_remove_last_fullstop = answer_extractor_remove_last_fullstop
        self.terminal_return_type = terminal_return_type
        self.generation_type = generation_type
        self.reset_queries_as_sentences = reset_queries_as_sentences
        self.max_num_sentences = max_num_sentences
        self.terminal_state_next_model = terminal_state_next_model
        self.shuffle_paras = shuffle_paras
        self.disable_exit = disable_exit
        self.question_prefix = question_prefix

        # Run 'python -m spacy download en_core_web_sm' if not downloaded already.
        self.spacy_object = spacy.load("en_core_web_sm")

    def return_model_calls(self):
        return {"step_by_step_cot": self.num_calls}

    def query(self, state, debug=False):

        exit_generation = False

        if f"generated_{self.generation_type}" not in state.data:
            state.data[f"generated_{self.generation_type}"] = []

        if len(state.data[f"generated_{self.generation_type}"]) >= self.max_num_sentences:
            exit_generation = True

        new_state = state.copy()
        return_answer = "EMPTY"
        return_titles = json.dumps(state.data["titles"])
        return_pids = json.dumps(
            [  # use this (^|v) as we don't want pinned to be part of returned titles/paras.
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(state.data["titles"], state.data["paras"])
            ]
        )

        # Don't bother wasting expensive llm call if we're already going to exist afterwards.
        if not exit_generation:

            question = state.data["question"]
            titles, paras = add_and_reorder_if_pinned(
                state.data["titles"],
                state.data["paras"],
                state.data["metadata"].get("pinned_title", None),
                state.data["metadata"].get("pinned_para", None),
                state.data["metadata"].get("pin_position", None),
            )
            zipped_titles_paras = list(zip(titles, paras))
            if self.shuffle_paras:
                random.shuffle(zipped_titles_paras)

            context = "\n\n".join(
                [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
            )
            generation_so_far = " ".join(state.data[f"generated_{self.generation_type}"])

            if self.question_prefix:
                assert self.question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
                question = self.question_prefix + question

            if self.add_context:
                test_example_str = context + "\n\n" + f"Q: {question}" + "\n" + f"A: {generation_so_far}"
            else:
                test_example_str = f"Q: {question}" + "\n" + f"A: {generation_so_far}"

            prompt = "\n\n\n".join([self.prompt, test_example_str]).strip()

            output_text_scores = self.generator.generate_text_sequence(prompt)
            if len(output_text_scores) > 1:
                print("Can not handle more than one answer for this model yet" + "\n" + str(output_text_scores))

            new_generation = output_text_scores[0][0].strip()
            new_sents = list(self.spacy_object(new_generation).sents)
            if new_sents:
                new_generation = new_sents[0].text
                new_state.data[f"generated_{self.generation_type}"].append(new_generation)

                if self.answer_extractor_regex.match(new_generation):
                    return_answer = self.answer_extractor_regex.match(new_generation).group(1)
                    if self.answer_extractor_remove_last_fullstop and return_answer.endswith("."):
                        return_answer = return_answer[:-1]
                    exit_generation = True

            else:
                if self.disable_exit:  # Add just empty sentence so exit controller can exit.
                    new_state.data[f"generated_{self.generation_type}"].append("")
                exit_generation = True

        if self.disable_exit:
            exit_generation = False

        if exit_generation:

            if self.terminal_return_type == "answer":  # answer
                output = return_answer
            elif self.terminal_return_type == "pids":  # pids
                output = return_pids
            else:  # titles
                assert self.terminal_return_type == "titles"
                output = return_titles

            if self.terminal_state_next_model is not None:
                new_state.next = self.terminal_state_next_model
            else:
                new_state.next = self.end_state

        else:

            # It should output full COT so far, not just what's generated in this round.
            output = " ".join(new_state.data[f"generated_{self.generation_type}"])
            new_state.next = self.next_model

        if self.reset_queries_as_sentences:
            # deepcopy is necessary
            new_state.data["generated_queries"] = copy.deepcopy(new_state.data["generated_sentences"])

        assert isinstance(output, str)
        new_state.data.add_answer(QuestionAnsweringStep(answer=output, score=0, participant=state.next))

        self.num_calls += 1

        return new_state


class StepByStepExitControllerParticipant(ParticipantModel):
    """
    Goes with StepByStepCOTGenParticipant and StepByStepLLMTitleGenParticipant
    and controls whether to exit or not.
    """

    def __init__(
        self,
        answer_extractor_regex=".* answer is (.*)",
        answer_extractor_remove_last_fullstop=True,
        terminal_return_type="titles",
        max_num_sentences=10,
        terminal_state_next_model=None,
        global_max_num_paras=100,
        generation_key="generated_sentences",
        next_model=None,
        end_state="[EOQ]",
    ):
        if terminal_return_type not in ("answer", "titles", "pids"):
            raise Exception(f"terminal_return_type has to be one of answer or titles. Found {terminal_return_type}.")

        self.num_calls = 0
        self.answer_extractor_regex = re.compile(answer_extractor_regex)
        self.answer_extractor_remove_last_fullstop = answer_extractor_remove_last_fullstop
        self.terminal_return_type = terminal_return_type
        self.max_num_sentences = max_num_sentences
        self.terminal_state_next_model = terminal_state_next_model
        self.global_max_num_paras = global_max_num_paras
        self.generation_key = generation_key
        self.next_model = next_model
        self.end_state = end_state

    def return_model_calls(self):
        return {"step_by_step_exit_controller": self.num_calls}

    def query(self, state, debug=False):

        if self.generation_key not in state.data:
            state.data[self.generation_key] = []
        generated_sentences = state.data[self.generation_key]

        new_state = state.copy()
        return_answer = "EMPTY"
        return_titles = json.dumps(state.data["titles"])
        return_pids = json.dumps(
            [  # keep using these as we don't want pinned to be part of returned titiles
                get_pid_for_title_paragraph_text(title, paragraph_text)
                for title, paragraph_text in zip(state.data["titles"], state.data["paras"])
            ]
        )

        assert_unique_titles_paras(state.data["titles"], state.data["paras"])
        assert len(state.data["paras"]) <= self.global_max_num_paras

        exit_generation = False

        if state.data[self.generation_key] and not state.data[self.generation_key][-1]:
            exit_generation = True

        if len(state.data[self.generation_key]) >= self.max_num_sentences:
            exit_generation = True

        # backup if regex doesn't match but we need to exit.
        if self.generation_key in ("generated_sub_answers", "generated_sub_questions"):
            return_answer = generated_sentences[-1]
        else:
            return_answer = " ".join(generated_sentences)

        if generated_sentences and self.answer_extractor_regex.match(generated_sentences[-1]):
            return_answer = self.answer_extractor_regex.match(generated_sentences[-1]).group(1)
            if self.answer_extractor_remove_last_fullstop and return_answer.endswith("."):
                return_answer = return_answer[:-1]
            exit_generation = True

        if exit_generation:

            if self.terminal_return_type == "answer":  # answer
                output = return_answer
            elif self.terminal_return_type == "pids":  # pids
                output = return_pids
            else:  # titles
                assert self.terminal_return_type == "titles"
                output = return_titles

            if self.terminal_state_next_model is not None:
                new_state.next = self.terminal_state_next_model
            else:
                new_state.next = self.end_state

        else:
            output = "Exit? No."
            new_state.next = self.next_model

        assert isinstance(output, str)
        new_state.data.add_answer(QuestionAnsweringStep(answer=output, score=0, participant=state.next))

        self.num_calls += 1

        return new_state
