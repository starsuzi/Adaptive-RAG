import json
from datetime import datetime
import random
import re

from dateutil.parser import parse

from commaqa.execution.llm_qa_model import LLMQAModel
from commaqa.inference.data_instances import QuestionAnsweringStep
from commaqa.inference.model_search import ParticipantModel
from commaqa.inference.ircot import para_to_text, add_and_reorder_if_pinned, get_spacy_object


random.seed(100)  # Don't change


def extract_key_information(state, info_type: str) -> str:
    if info_type is None:
        return ""

    elif info_type == "cot":
        generated_sentences = state.data["generated_sentences"]
        generated_sentences = [
            sentence.strip() for sentence in generated_sentences if "answer is".lower() not in sentence.lower()
        ]
        key_info_text = " ".join(generated_sentences).strip()
        key_info_text = "\n\nKey Information: " + key_info_text
        return key_info_text

    elif info_type == "subqas":
        raise Exception("Make sure this is really what you want. In subqa format, the last step is still cot.")
        generated_sentences = state.data["generated_sentences"]
        generated_sentences = [
            sentence.strip() for sentence in generated_sentences if "answer is".lower() not in sentence.lower()
        ]
        key_info_text = "\n".join([e.strip() for e in generated_sentences])
        key_info_text = "\n\nKey Information:\n" + key_info_text
        return key_info_text

    else:
        raise Exception(f"Unknown info_type {info_type}")


class LLMQAParticipantModel(ParticipantModel):
    def __init__(
        self,
        next_model=None,
        end_state="[EOQ]",
        extractor_regex=None,
        extractor_remove_last_fullstop=False,
        allow_empty_answers=True,
        max_para_num_words=350,
        shuffle_paras=False,
        answer_is_numbered_list=False,
        store_sents_in_generated_sentences=False,
        question_prefix="",
        key_info_type=None,
        **kwargs,
    ):
        self.answer_is_numbered_list = answer_is_numbered_list

        if answer_is_numbered_list:
            kwargs["stop"] = ["\n\n"]

        self.key_info_type = key_info_type
        self.qa_model = LLMQAModel(**kwargs)
        self.next_model = next_model
        self.end_state = end_state
        self.extractor_regex = None
        if extractor_regex is not None:
            self.extractor_regex = re.compile(extractor_regex)
        self.extractor_remove_last_fullstop = extractor_remove_last_fullstop
        self.num_calls = 0
        self.max_para_num_words = max_para_num_words
        self.allow_empty_answers = allow_empty_answers
        self.shuffle_paras = shuffle_paras
        self.question_prefix = question_prefix  # Don't strip this. It may have \n
        self.store_sents_in_generated_sentences = store_sents_in_generated_sentences

    def return_model_calls(self):
        return {"llm_qa": self.num_calls}

    def update_state(self, answer, state):
        if not self.allow_empty_answers and answer == "":
            print("WARNING: Generate empty answer.")
            return []
        new_state = state.copy()
        new_state.data.add_answer(QuestionAnsweringStep(answer=json.dumps(answer), score=0, participant=state.next))
        new_state.next = self.next_model if self.next_model else self.end_state
        return new_state

    def query(self, state, debug=False):
        question = state.data.get_last_question()

        if self.question_prefix:
            assert self.question_prefix.endswith("\n") or self.question_prefix.endswith(" ")
            question = self.question_prefix + question

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

        if "paras" in state.data:
            context = [para_to_text(title, para, self.max_para_num_words) for title, para in zipped_titles_paras]
        else:
            context = ""

        self.num_calls += 1
        context_suffix = extract_key_information(state, self.key_info_type)
        answer, facts_used = self.qa_model.ask_question(
            input_question=question, context=context, context_suffix=context_suffix
        )

        if self.extractor_regex and isinstance(answer, str) and self.extractor_regex.match(answer):
            answer = self.extractor_regex.match(answer).group(1)
            if self.extractor_remove_last_fullstop and answer.endswith("."):
                answer = answer[:-1]

        if self.answer_is_numbered_list:
            answer = [re.sub(r"^\d+\.", "", e.strip()).strip() for e in str(answer).split("\n")]
            answer = [e for e in answer if e]
            answer = list(dict.fromkeys(answer).keys())

        if self.store_sents_in_generated_sentences:
            spacy_object = get_spacy_object()
            state.data["generated_sentences"] = [sent.text_with_ws for sent in spacy_object(answer).sents]

        return self.update_state(answer=answer, state=state)


def date_difference(date1: str, date2: str, units: str = "years"):
    default_date = datetime(3000, 1, 1)
    try:
        date1_datetime = parse(date1, default=default_date)
        date2_datetime = parse(date2, default=default_date)
    except Exception:
        # couldn't parse date
        return None
    # if one doesn't have month set, not usable
    if date1_datetime.year == default_date.year and date1_datetime.month == default_date.month:
        return None
    if date2_datetime.year == default_date.year and date2_datetime.month == default_date.month:
        return None

    if date1_datetime.year == default_date.year and date2_datetime.year != default_date.year:
        # one date is relative and other is not
        date1_datetime = date1_datetime.replace(year=date2_datetime.year)
    elif date2_datetime.year == default_date.year and date1_datetime.year != default_date.year:
        # one date is relative and other is not
        date2_datetime = date2_datetime.replace(year=date1_datetime.year)

    if units == "days":
        return (date1_datetime - date2_datetime).days
    if units == "months":
        return (date1_datetime.year - date2_datetime.year) * 12 + (date1_datetime.month - date2_datetime.month)
    if units == "years":
        # human annotations are often on just the year value
        return date1_datetime.year - date2_datetime.year
    print("Unknown unit:" + units)
    return None


def sort_without_duplicates(arr):
    last_val = None
    output_arr = []
    for (key, val) in sorted(arr, key=lambda x: x[1]):
        if val == last_val:
            continue
        else:
            output_arr.append((key, val))
            last_val = val
    return output_arr


def sorted_key(arr):
    return [x[0] for x in sort_without_duplicates(arr)]


def sorted_value(arr):
    return [x[1] for x in sort_without_duplicates(arr)]
