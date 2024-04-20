import logging
import re

from commaqa.configs.utils import execute_steps
from commaqa.dataset.utils import get_predicate_args, align_assignments, get_question_indices, valid_answer, NOANSWER
from commaqa.execution.constants import KBLOOKUP_MODEL
from commaqa.execution.operation_executer import OperationExecuter

logger = logging.getLogger(__name__)


class ModelExecutor:
    def __init__(self, predicate_language, model_name, kblookup, ignore_input_mismatch=False):
        self.predicate_language = predicate_language
        self.model_name = model_name
        self.kblookup = kblookup
        self.ignore_input_mismatch = ignore_input_mismatch
        self.num_calls = 0

    def find_qpred_assignments(self, input_question, question_definition):
        question_re = re.escape(question_definition)
        varid_groupid = {}
        qindices = get_question_indices(question_definition)
        for num in qindices:
            question_re = question_re.replace("\\$" + str(num), "(?P<G" + str(num) + ">.+)")
            varid_groupid["$" + str(num)] = "G" + str(num)

        qmatch = re.match(question_re, input_question)
        if qmatch:
            assignments = {}
            for varid, groupid in varid_groupid.items():
                assignments[varid] = qmatch.group(groupid)
            return assignments
        return None

    def ask_question(self, input_question, context=None):
        self.num_calls += 1
        qpred, qargs = get_predicate_args(input_question)
        if context:
            raise ValueError(
                "Input context passed to ModelExecutor which does not use context!" + "\n{}".format(context)
            )
        if qpred is not None:
            return self.ask_question_predicate(question_predicate=input_question)
        else:
            answers, facts_used = None, None
            for pred_lang in self.predicate_language:
                for question in pred_lang.questions:
                    assignments = self.find_qpred_assignments(
                        input_question=input_question, question_definition=question
                    )
                    if assignments is not None:
                        new_pred = pred_lang.predicate
                        for varid, assignment in assignments.items():
                            new_pred = new_pred.replace(varid, assignment)
                        answers, facts_used = self.ask_question_predicate(new_pred)
                        if valid_answer(answers):
                            # if this is valid answer, return it
                            return answers, facts_used

            # if answers is not None:
            #     # some match found for the question but no valid answer.
            #     # Return the last matching answer.
            #     return answers, facts_used
            if not self.ignore_input_mismatch:
                raise ValueError(
                    "No matching question found for {} "
                    "in pred_lang:\n{}".format(input_question, self.predicate_language)
                )
            else:
                # no matching question. return NOANSWER
                return NOANSWER, []

    def ask_question_predicate(self, question_predicate):
        qpred, qargs = get_predicate_args(question_predicate)
        for pred_lang in self.predicate_language:
            mpred, margs = get_predicate_args(pred_lang.predicate)
            if mpred != qpred:
                continue
            if pred_lang.steps:
                model_library = {KBLOOKUP_MODEL: self.kblookup}
                kb_executor = OperationExecuter(model_library)
                source_assignments = {x: x for x in qargs}
                curr_assignment, assignment_map = align_assignments(
                    target_predicate=pred_lang.predicate,
                    source_predicate=question_predicate,
                    source_assignments=source_assignments,
                )
                assignments = execute_steps(
                    steps=pred_lang.steps,
                    input_assignments=curr_assignment,
                    executer=kb_executor,
                    pred_lang_config=None,
                    input_model=KBLOOKUP_MODEL,
                )

                if assignments:
                    last_answer = pred_lang.steps[-1].answer
                    return assignments[last_answer], assignments["facts_used"]
                elif assignments is None:
                    # execution failed, try next predicate
                    continue
                else:
                    logger.debug("No answer found for question: {}".format(question_predicate))
                    return [], []
            else:
                return self.kblookup.ask_question_predicate(question_predicate)
        # No match found for predicate
        error = "No matching predicate for {}".format(question_predicate)
        if self.ignore_input_mismatch:
            logger.debug(error)
            return NOANSWER, []
        else:
            raise ValueError(error)
