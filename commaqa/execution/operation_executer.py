import json
import logging

from commaqa.dataset.utils import flatten_list, get_answer_indices, NOANSWER, valid_answer

logger = logging.getLogger(__name__)


class OperationExecuter:
    def __init__(self, model_library, ignore_input_mismatch=False):
        self.model_library = model_library
        self.ignore_input_mismatch = ignore_input_mismatch
        self.num_calls = 0

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

    def execute_select(self, operation, model, question, assignments, context=None):
        indices = get_answer_indices(question)
        for index in indices:
            idx_str = "#" + str(index)
            if idx_str not in assignments:
                raise ValueError(
                    "SELECT: Can not perform select operation with input arg: {}" " No assignments yet!".format(idx_str)
                )
            question = question.replace(idx_str, json.dumps(assignments[idx_str]))
        answers, facts_used = self.model_library[model].ask_question(question, context)
        if not valid_answer(answers):
            return NOANSWER, []
        answers = self.execute_sub_operations(answers, operation)
        return answers, facts_used

    def execute_project(self, operation, model, question, assignments, context=None):
        indices = get_answer_indices(question)
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
        answers = []
        facts_used = []
        operation_seq = operation.split("_")
        first_op = operation_seq[0]
        if not isinstance(assignments[idx_str], list):
            raise ValueError(
                "PROJECT: Can not perform project operation on a non-list input: {}"
                " Operation: {} Question: {}".format(assignments[idx_str], operation, question)
            )
        for item in assignments[idx_str]:
            # print(idx_str, item, assignments)
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
                if not isinstance(item, str):
                    raise ValueError(
                        "PROJECT: Item: {} is not a string in assigments: {}. "
                        "Expected for project".format(item, assignments[idx_str])
                    )
                new_question = question.replace(idx_str, item)
            curr_answers, curr_facts = self.model_library[model].ask_question(new_question, context)
            if not valid_answer(curr_answers):
                return NOANSWER, []
            facts_used.extend(curr_facts)

            if first_op == "projectValues":
                answers.append((item[0], curr_answers))
            elif first_op == "projectKeys":
                answers.append((curr_answers, item[1]))
            elif first_op == "project":
                answers.append((item, curr_answers))

        answers = self.execute_sub_operations(answers, operation)
        return answers, facts_used

    def execute_filter(self, operation, model, question, assignments, context=None):
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
        answers = []
        facts_used = []
        operation_seq = operation.split("_")
        first_op = operation_seq[0]
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
                if not isinstance(item, str):
                    raise ValueError(
                        "FILTER: Item: {} is not a string in assigments: {}. "
                        "Expected for filter".format(item, assignments[idx_str])
                    )
                new_question = question.replace(idx_str, item)

            answer, curr_facts = self.model_library[model].ask_question(new_question, context)
            if not valid_answer(answer):
                return NOANSWER, []
            if not isinstance(answer, str):
                raise ValueError(
                    "FILTER: Incorrect question type for filter. Returned answer: {}"
                    " for question: {}".format(answer, new_question)
                )
            answer = answer.lower()
            if answer == "yes" or answer == "1" or answer == "true":
                answers.append(item)
            facts_used.extend(curr_facts)
        answers = self.execute_sub_operations(answers, operation)
        return answers, facts_used

    def execute_operation(self, operation, model, question, assignments, context=None):
        self.num_calls += 1
        if model not in self.model_library:
            error_mesg = "Model: {} not found in " "model_library: {}".format(model, self.model_library.keys())
            if not self.ignore_input_mismatch:
                raise ValueError(error_mesg)
            else:
                logger.debug(error_mesg)
                return NOANSWER, []
        try:
            if operation.startswith("select"):
                return self.execute_select(operation, model, question, assignments, context)
            elif operation.startswith("project"):
                return self.execute_project(operation, model, question, assignments, context)
            elif operation.startswith("filter"):
                return self.execute_filter(operation, model, question, assignments, context)
            else:
                logger.debug("Can not execute operation: {}. Returning empty list".format(operation))
                return NOANSWER, []
        except ValueError as e:
            if self.ignore_input_mismatch:
                logger.debug(
                    "Can not execute operation: {} question: {} "
                    "with assignments: {}".format(operation, question, assignments)
                )
                logger.debug(str(e))
                return NOANSWER, []
            else:
                raise e
