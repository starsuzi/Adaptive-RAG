import json
import logging
import random
import string
from typing import Dict, List

from commaqa.configs.step_config import StepConfig
from commaqa.configs.utils import execute_steps
from commaqa.dataset.utils import dict_product, align_assignments, nonempty_answer, is_question_var
from commaqa.execution.model_executer import ModelExecutor
from commaqa.execution.operation_executer import OperationExecuter

logger = logging.getLogger(__name__)


class TheoryConfig:
    def __init__(self, theory_json):
        self.steps = [StepConfig(x) for x in theory_json["steps"]]
        self.questions = theory_json.get("questions")
        self.init = theory_json["init"]

    def to_json(self):
        return {"steps": [x.to_json() for x in self.steps], "questions": self.questions, "init": self.init}

    def to_str(self):
        return json.dumps(self.to_json())

    def get_possible_assignments(
        self, entities: Dict[str, List[str]], model_library: Dict[str, ModelExecutor], pred_lang_config
    ):
        assignment_dict = {}
        for key, ent_type in self.init.items():
            assignment_dict[key] = entities[ent_type]
        possible_assignments = dict_product(assignment_dict)
        # assume no duplicates in assignments
        possible_assignments = [
            assignment
            for assignment in possible_assignments
            if len(set(assignment.values())) == len(assignment.values())
        ]
        op_executor = OperationExecuter(model_library=model_library)
        output_assignments = []
        for curr_assignment in possible_assignments:
            # print(self.to_json())
            new_assignment = execute_steps(
                steps=self.steps,
                input_assignments=curr_assignment,
                executer=op_executor,
                pred_lang_config=pred_lang_config,
                input_model=None,
            )
            if new_assignment:
                output_assignments.append(new_assignment)

        if len(output_assignments) < 2:
            logger.debug(
                "Few assignments: {} found for theory: {} given kb:\n {}".format(
                    json.dumps(output_assignments, indent=2),
                    self.to_str(),
                    json.dumps(list(model_library.values())[0].kblookup.kb, indent=2),
                )
            )
        return output_assignments

    def create_decompositions(self, pred_lang_config, assignment):
        decomposition = []
        for step in self.steps:
            valid_configs = pred_lang_config.find_valid_configs(step.question)
            if len(valid_configs) == 0:
                raise ValueError("No predicate config matches {}".format(step.question))

            #     # model less operation
            #     model = "N/A"
            #     print(step.question)
            #     question = step.question
            #     for k, v in assignment.items():
            #         if k.startswith("$"):
            #             question = question.replace(k, v)
            # else:
            lang_conf = random.choice(valid_configs)
            model = lang_conf.model
            question = random.choice(lang_conf.questions)
            _, assignment_map = align_assignments(lang_conf.predicate, step.question, assignment)
            for lang_pred_arg, question_pred_arg in assignment_map.items():
                if is_question_var(question_pred_arg):
                    question = question.replace(lang_pred_arg, assignment[question_pred_arg])
                else:
                    # replace the question idx with the appropriate answer idx in the theory
                    question = question.replace(lang_pred_arg, question_pred_arg)
            answer = assignment[step.answer]
            decomposition.append({"m": model, "q": question, "a": answer, "op": step.operation})
        return decomposition

    def create_questions(self, entities: Dict[str, List[str]], pred_lang_config, model_library):
        possible_assignments = self.get_possible_assignments(
            entities=entities, pred_lang_config=pred_lang_config, model_library=model_library
        )
        qa = []
        for assignment in possible_assignments:
            decomposition = self.create_decompositions(pred_lang_config=pred_lang_config, assignment=assignment)
            # move facts_used out of the assignment structure
            facts_used = list(set(assignment["facts_used"]))
            del assignment["facts_used"]
            question = random.choice(self.questions)
            answer = assignment[self.steps[-1].answer]
            for p, f in assignment.items():
                if p in question:
                    question = question.replace(p, f)
            if decomposition[-1]["a"] != answer:
                raise ValueError(
                    "Answer to the last question in decomposition not the same as the "
                    "final answer!.\n Decomposition:{} \n Question: {} \n Answer: {}"
                    "".format(decomposition, question, answer)
                )
            # ignore questions with no valid answers
            if not nonempty_answer(answer):
                continue
            # ignore questions with too many answers
            if isinstance(answer, list) and len(answer) > 5:
                continue
            qa.append(
                {
                    "question": question,
                    "answer": answer,
                    "assignment": assignment,
                    "config": self.to_json(),
                    "decomposition": decomposition,
                    "facts_used": facts_used,
                    "id": "".join([random.choice(string.hexdigits) for n in range(16)]).lower(),
                }
            )
        return qa
