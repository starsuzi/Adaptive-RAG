from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class BasicDataInstance(dict):
    _REQUIRED_ATTRS = set([])

    def __init__(self, input_data):
        dict.__init__({})
        self.update(input_data)
        for item in type(self)._REQUIRED_ATTRS:
            if item not in self:
                self[item] = []


class QuestionGeneratorData(BasicDataInstance):
    _REQUIRED_ATTRS = set(
        [
            "question_seq",
            "subquestion_seq",
            "answer_seq",
            "command_seq",
            "model_seq",
            "operation_seq",
            "score_seq",
            "para_seq",
        ]
    )


@dataclass
class InferenceStep:
    score: float
    participant: str


@dataclass
class QuestionGenerationStep(InferenceStep):
    question: str


@dataclass
class QuestionParsingStep(InferenceStep):
    operation: str
    model: str
    subquestion: str


@dataclass
class QuestionAnsweringStep(InferenceStep):
    answer: str


@dataclass
class AnswerSubOperationStep(InferenceStep):
    sub_operation: str
    input_answer: Any
    output_answer: Any


@dataclass
class Task:
    task_question: QuestionGenerationStep
    task_participant: str


class StructuredDataInstance(BasicDataInstance):
    _REQUIRED_ATTRS = set(["inference_seq"])

    def __init__(self, input_data):
        super().__init__(input_data)
        self.inference_ref_stack = [self]
        self.task_stack = []

    def add_answer(self, qastep: QuestionAnsweringStep):
        self.get_current_inference_seq().append(qastep)

    def add_qgen(self, qgenstep: QuestionGenerationStep):
        self.get_current_inference_seq().append(qgenstep)

    def add_qparse(self, qgenstep: QuestionParsingStep):
        self.get_current_inference_seq().append(qgenstep)

    def add_suboperation_step(self, subop_step: AnswerSubOperationStep):
        self.get_current_inference_seq().append(subop_step)

    def add_subdecomp(self, subdata: StructuredDataInstance):
        self.get_current_inference_seq().append(subdata)
        self.inference_ref_stack.append(subdata)

    def add_task(self, task: Task):
        self.task_stack.append(task)

    def pop_task(self):
        return self.task_stack.pop()

    def has_tasks(self):
        return len(self.task_stack) > 0

    def get_last_step(self):
        return self.get_current_inference_seq()[-1]

    def get_last_generator(self):
        for step in reversed(self.get_current_inference_seq()):
            if isinstance(step, QuestionGenerationStep):
                return step.participant

    def popup_decomp_level(self):
        if len(self.inference_ref_stack) == 1:
            raise ValueError(
                "Will pop up to an empty inference stack!\n{}".format(self.get_printable_reasoning_chain())
            )
        self.inference_ref_stack.pop()

    def at_root_level(self):
        return len(self.inference_ref_stack) == 1

    def get_current_inference_seq(self):
        return self.inference_ref_stack[-1]["inference_seq"]

    def get_current_inference_data(self):
        return self.inference_ref_stack[-1]

    def get_current_aseq(self):
        aseq = []
        for step in self.get_current_inference_seq():
            if isinstance(step, QuestionAnsweringStep):
                aseq.append(step.answer)

        return aseq

    def get_current_qseq(self):
        qseq = []
        for step in self.get_current_inference_seq():
            if isinstance(step, QuestionGenerationStep):
                qseq.append(step.question)

        return qseq

    def get_current_subqseq(self):
        qseq = []
        for step in self.get_current_inference_seq():
            if isinstance(step, QuestionParsingStep):
                qseq.append(step.subquestion)

        return qseq

    def get_last_question(self):
        for step in reversed(self.get_current_inference_seq()):
            if isinstance(step, QuestionGenerationStep):
                return step.question
        raise ValueError("No last question! No question generated yet")

    def get_last_answer(self):
        for step in reversed(self.get_current_inference_seq()):
            if isinstance(step, QuestionAnsweringStep):
                return step.answer
        raise ValueError("No last answer! No answer generated yet")

    def get_printable_reasoning_chain(self, indent=""):
        # chain = ": {}".format(self["qid"], self["question"])
        chain = ""
        for step in self["inference_seq"]:
            if isinstance(step, AnswerSubOperationStep):
                if chain:
                    chain += "\n"
                chain += indent + "O: {}({}) ==> {}".format(
                    step.sub_operation, json.dumps(step.input_answer), json.dumps(step.output_answer)
                )
            if isinstance(step, QuestionGenerationStep):
                if chain:
                    chain += "\n"
                chain += indent + "Q: " + step.question
            if isinstance(step, QuestionAnsweringStep):
                if chain:
                    chain += "\n"
                chain += indent + "A: " + step.answer
            if isinstance(step, StructuredDataInstance):
                if chain:
                    chain += "\n"
                chain += step.get_printable_reasoning_chain(indent=indent + "    ")
        return chain

    def get_last_question_generator(self):
        for step in reversed(self["inference_seq"]):
            if isinstance(step, QuestionGenerationStep):
                return step.participant
        raise ValueError("No last question! No question generated yet")
