import logging
from copy import deepcopy
from typing import List, Dict

from commaqa.configs.predicate_language_config import PredicateLanguageConfig
from commaqa.configs.step_config import StepConfig
from commaqa.dataset.utils import is_question_var, nonempty_answer
from commaqa.execution.operation_executer import OperationExecuter

logger = logging.getLogger(__name__)


def execute_steps(
    steps: List[StepConfig],
    input_assignments: Dict[str, str],
    executer: OperationExecuter,
    pred_lang_config: PredicateLanguageConfig = None,
    input_model: str = None,
):
    curr_assignment = deepcopy(input_assignments)
    if "facts_used" not in curr_assignment:
        curr_assignment["facts_used"] = []

    for step in steps:
        if input_model is None:
            model = pred_lang_config.find_model(step.question)
            if model is None:
                raise ValueError("No model found for {}".format(step.question))
        else:
            model = input_model

        new_question = step.question
        for k, v in curr_assignment.items():
            # only replace question variables($1, $2). Answer variables (#1, #2) used by executer
            if is_question_var(k):
                new_question = new_question.replace(k, v)
        answers, curr_facts = executer.execute_operation(
            operation=step.operation, model=model, question=new_question, assignments=curr_assignment
        )
        if answers is None:
            # execution failed
            return None
        elif nonempty_answer(answers):
            curr_assignment[step.answer] = answers
            curr_assignment["facts_used"].extend(curr_facts)
        else:
            logger.debug(
                "Stopped Execution. Empty answer: {}\n"
                "Question: {}\n Step: {}\n Assignment: {}".format(
                    answers, new_question, step.to_json(), curr_assignment
                )
            )
            return {}
    return curr_assignment
