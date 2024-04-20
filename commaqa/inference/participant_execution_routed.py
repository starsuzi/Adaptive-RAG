import json
import logging
import re

from commaqa.dataset.utils import get_answer_indices, flatten_list
from commaqa.inference.data_instances import (
    Task,
    QuestionGenerationStep,
    AnswerSubOperationStep,
    QuestionAnsweringStep,
    StructuredDataInstance,
    QuestionParsingStep,
)
from commaqa.inference.model_search import ParticipantModel

logger = logging.getLogger(__name__)


class RoutedExecutionParticipant(ParticipantModel):
    def __init__(self, next_model=None, end_state="[EOQ]"):
        self.next_model = next_model
        self.end_state = end_state
        self.per_model_calls = {"executer": 0, "op_executer": 0}
        self.op_model_q_regex = re.compile("\((.+)\) \[([^\]]+)\] (.*)")
        self.operation_only_regex = re.compile("\((.+)\)(.*)")

    def return_model_calls(self):
        return self.per_model_calls

    def query(self, state, debug=False):
        """The main function that interfaces with the overall search and
        model controller, and manipulates the incoming data.

        :param state: the state of controller and model flow.
        :type state: launchpadqa.question_search.model_search.SearchState
        :rtype: list
        """
        ## the data
        data = state._data

        question = data.get_last_question()
        m = self.op_model_q_regex.match(question)
        if m is not None:
            try:
                return self.add_model_questions(m, state)
            except ValueError as e:
                logger.debug("Failed execution {}".format(data.get_printable_reasoning_chain()))
                if debug:
                    raise e
                return []
        else:
            m = self.operation_only_regex.match(question)
            if m is not None:
                try:
                    return self.execute_operation(operation=m.group(1), arguments=m.group(2), state=state.copy())
                except Exception as e:
                    logger.debug("Failed execution {}".format(data.get_printable_reasoning_chain()))
                    if debug:
                        raise e
                    return []
            else:
                logger.debug("No match for {}".format(question))
                return []

    def add_model_questions(self, m, state):
        self.per_model_calls["executer"] += 1
        data = state.data
        question = data.get_last_question()
        assignment = {}
        for ans_idx, ans in enumerate(data.get_current_aseq()):
            assignment["#" + str(ans_idx + 1)] = json.loads(ans)
        if "paras" in data and len(data["paras"]) > 0:
            assignment["#C"] = ["PARA_" + str(x) for x in range(len(data["paras"]))]
        operation = m.group(1)
        model = m.group(2)
        sub_question = m.group(3)
        new_state = state.copy()
        new_state.data.add_qparse(
            QuestionParsingStep(
                score=0, participant=state.next, operation=operation, model=model, subquestion=sub_question
            )
        )
        new_state.data.add_subdecomp(
            StructuredDataInstance(input_data={"qid": state.data["qid"], "query": question, "question": question})
        )
        if operation.startswith("select"):
            new_state = self.get_select_state(
                question=sub_question, model=model, operation=operation, assignments=assignment, state=new_state
            )
        elif operation.startswith("project"):
            new_state = self.get_project_state(
                question=sub_question, model=model, operation=operation, assignments=assignment, state=new_state
            )
        elif operation.startswith("filter"):
            new_state = self.get_filter_state(
                question=sub_question, model=model, operation=operation, assignments=assignment, state=new_state
            )
        else:
            raise ValueError("Not implemented! " + operation)
        new_state.next = self.end_state
        return new_state

    def get_select_state(self, question, model, operation, assignments, state):
        indices = get_answer_indices(question)
        for index in indices:
            idx_str = "#" + str(index)
            if idx_str not in assignments:
                raise ValueError(
                    "SELECT: Can not perform select operation with input arg: {}" " No assignments yet!".format(idx_str)
                )
            question = question.replace(idx_str, json.dumps(assignments[idx_str]))
        # add tasks in reverse order so they are popped correctly

        # merge
        merge_question = "(" + operation + ")"
        state.data.add_task(
            Task(
                task_question=QuestionGenerationStep(score=0, participant=state.next, question=merge_question),
                task_participant=state.next,
            )
        )

        state.data.add_task(
            Task(
                task_question=QuestionGenerationStep(score=0, participant=state.next, question=question),
                task_participant=model,
            )
        )

        return state

    def get_project_state(self, question, model, operation, assignments, state):
        indices = get_answer_indices(question)
        if "#C" in question and len(indices) == 0:
            indices = ["C"]

        if len(indices) > 1:
            raise ValueError(
                "PROJECT: Can not handle more than one answer idx: {} " "for project: {}".format(indices, question)
            )
        if len(indices) == 0:
            raise ValueError("PROJECT: Did not find any indices to project on " + str(question))
        idx_str = "#" + str(indices[0])
        if idx_str not in assignments:
            raise ValueError(
                "PROJECT: Can not perform project operation with input arg: {}" " No assignments yet!".format(idx_str)
            )

        operation_seq = operation.split("_")
        first_op = operation_seq[0]
        if not isinstance(assignments[idx_str], list):
            raise ValueError(
                "PROJECT: Can not perform project operation on a non-list input: {}"
                " Operation: {} Question: {}".format(assignments[idx_str], operation, question)
            )
        new_questions = []
        for item in assignments[idx_str]:
            if isinstance(item, list) and len(item) == 2:
                item = tuple(item)
            if first_op == "projectValues":
                # item should be a tuple
                if not isinstance(item, tuple):
                    raise ValueError(
                        "PROJECT: Item: {} is not a tuple in assignments: {}. "
                        "Expected for projectValues".format(item, assignments[idx_str])
                    )
                new_question = question.replace(idx_str, json.dumps(item[1]))
            elif first_op == "projectKeys":
                # item should be a tuple
                if not isinstance(item, tuple):
                    raise ValueError(
                        "PROJECT: Item: {} is not a tuple in assignments: {}. "
                        "Expected for projectKeys".format(item, assignments[idx_str])
                    )
                new_question = question.replace(idx_str, json.dumps(item[0]))
            else:
                if isinstance(item, int) or isinstance(item, float):
                    item = str(item)
                if not isinstance(item, str):
                    raise ValueError(
                        "PROJECT: Item: {} is not a string in assignments: {}. "
                        "Expected for project".format(item, assignments[idx_str])
                    )
                new_question = question.replace(idx_str, item)
            new_questions.append(new_question)

        # add tasks in reverse order so they are popped correctly
        merge_question = "(" + operation + ") " + json.dumps(assignments[idx_str])
        state.data.add_task(
            Task(
                task_question=QuestionGenerationStep(score=0, participant=state.next, question=merge_question),
                task_participant=state.next,
            )
        )

        for q in reversed(new_questions):
            state.data.add_task(
                Task(
                    task_question=QuestionGenerationStep(score=0, participant=state.next, question=q),
                    task_participant=model,
                )
            )
        return state

    def get_filter_state(self, question, model, operation, assignments, state):
        q_answer_indices = get_answer_indices(question)
        if len(q_answer_indices) > 1:
            # more than one answer index in the question, use the operation to identify the
            # answer idx to operate over
            op_answer_indices = get_answer_indices(operation)
            if len(op_answer_indices) != 1:
                raise ValueError(
                    "Need one answer idx to be specified in filter operation since "
                    "multiple specified in the question!  "
                    "Operation: {} Question: {}".format(operation, question)
                )
            else:
                operation_idx = op_answer_indices[0]
                for idx in q_answer_indices:
                    if idx != operation_idx:
                        # modify question directly to include the non-operated answer id
                        idx_str = "#" + str(idx)
                        if idx_str not in assignments:
                            raise ValueError(
                                "FILTER: Can not perform filter operation with input arg: {} "
                                "No assignments yet!".format(idx_str)
                            )
                        # print(question, idx_str, assignments)
                        question = question.replace(idx_str, json.dumps(assignments[idx_str]))
        elif len(q_answer_indices) == 1:
            operation_idx = q_answer_indices[0]
        else:
            raise ValueError(
                "FILTER: No answer index in question for filter"
                "Operation: {} Question: {}".format(operation, question)
            )

        idx_str = "#" + str(operation_idx)
        if idx_str not in assignments:
            raise ValueError(
                "FILTER: Can not perform filter operation with input arg: {}" " No assignments yet!".format(idx_str)
            )
        if not isinstance(assignments[idx_str], list):
            raise ValueError(
                "FILTER: Can not perform filter operation on a non-list input: {}"
                " Operation: {} Question: {}".format(assignments[idx_str], operation, question)
            )

        operation_seq = operation.split("_")
        first_op = operation_seq[0]
        new_questions = []
        for item in assignments[idx_str]:
            if isinstance(item, list) and len(item) == 2:
                item = tuple(item)
            if first_op.startswith("filterValues"):
                # item should be a tuple
                if not isinstance(item, tuple):
                    raise ValueError(
                        "FILTER: Item: {} is not a tuple in assignments: {}. "
                        "Expected for filterValues".format(item, assignments[idx_str])
                    )
                (key, value) = item
                new_question = question.replace(idx_str, json.dumps(value))
            elif first_op.startswith("filterKeys"):
                if not isinstance(item, tuple):
                    raise ValueError(
                        "FILTER: Item: {} is not a tuple in assignments: {}. "
                        "Expected for filterKeys".format(item, assignments[idx_str])
                    )
                (key, value) = item
                new_question = question.replace(idx_str, json.dumps(key))
            else:
                if isinstance(item, int) or isinstance(item, float):
                    item = str(item)
                if not isinstance(item, str):
                    raise ValueError(
                        "FILTER: Item: {} is not a string in assigments: {}. "
                        "Expected for filter".format(item, assignments[idx_str])
                    )
                new_question = question.replace(idx_str, item)
            new_questions.append(new_question)
            print(new_question)

        # add tasks in reverse order so they are popped correctly
        merge_question = "(" + operation + ") " + json.dumps(assignments[idx_str])
        state.data.add_task(
            Task(
                task_question=QuestionGenerationStep(score=0, participant=state.next, question=merge_question),
                task_participant=state.next,
            )
        )

        for q in reversed(new_questions):
            state.data.add_task(
                Task(
                    task_question=QuestionGenerationStep(score=0, participant=state.next, question=q),
                    task_participant=model,
                )
            )
        return state

    def is_true(self, ans):
        if isinstance(ans, bool):
            return ans
        elif isinstance(ans, str):
            answer = ans.lower()
            return answer == "yes" or answer == "1" or answer == "true"
        else:
            raise ValueError("Can't verify truth value for {}".format(ans))

    def execute_operation(self, operation, arguments, state):
        self.per_model_calls["op_executer"] += 1
        operation_seq = operation.split("_")
        # ignore any argument e.g. filter(#2)
        first_op = operation_seq[0].split("(")[0]
        answer_seq = [json.loads(a) for a in state.data.get_current_aseq()]
        if arguments:
            parsed_arguments = json.loads(arguments)
            assert len(answer_seq) == len(
                parsed_arguments
            ), "Answer_seq is not of the same length as parsed_args\n{}\n{}".format(answer_seq, parsed_arguments)
        if first_op == "projectValues":
            answers = list(zip([x[0] for x in parsed_arguments], answer_seq))
        elif first_op == "projectKeys":
            answers = list(zip(answer_seq, [x[1] for x in parsed_arguments]))
        elif first_op == "project":
            answers = list(zip(parsed_arguments, answer_seq))
        elif first_op == "select":
            assert len(answer_seq) == 1
            answers = answer_seq[0]
        elif first_op == "filter":
            answers = [x[0] for x in zip(parsed_arguments, answer_seq) if self.is_true(x[1])]
        else:
            raise ValueError("Not implemented! " + first_op)
        new_answers = self.execute_sub_operations(answers=answers, operation=operation)
        state.data.add_suboperation_step(
            AnswerSubOperationStep(
                score=0,
                participant=state.next,
                sub_operation=operation,
                input_answer=answers,
                output_answer=new_answers,
            )
        )
        state.data.add_answer(QuestionAnsweringStep(answer=json.dumps(new_answers), score=0, participant=state.next))
        state.data.popup_decomp_level()
        state.data.add_answer(QuestionAnsweringStep(answer=json.dumps(new_answers), score=0, participant=state.next))
        state.next = self.end_state
        return state

    def execute_sub_operations(self, answers, operation):
        operation_seq = operation.split("_")
        for op in operation_seq[1:]:
            if op == "flat":
                answers = flatten_list(answers)
            elif op == "unique":
                if not isinstance(answers, list):
                    raise ValueError("SUBOP: unique can only be applied to list. " "Input: {}".format(answers))
                seen_objs = set()
                output_answers = []
                for item in answers:
                    # handle any structure: convert to str
                    item_str = json.dumps(item)
                    if item_str not in seen_objs:
                        output_answers.append(item)
                        seen_objs.add(item_str)
                answers = output_answers
            elif op == "keys":
                answers = [x[0] for x in answers]
            elif op == "values":
                answers = [x[1] for x in answers]
            else:
                raise ValueError("SUBOP: Unknown sub-operation: {}".format(op))
        return answers
