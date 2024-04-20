import json
import logging
import re

from commaqa.configs.predicate_language_config import ModelQuestionConfig
from commaqa.dataset.utils import valid_answer, nonempty_answer
from commaqa.execution.operation_executer import OperationExecuter
from commaqa.execution.utils import build_models
from commaqa.inference.data_instances import QuestionAnsweringStep
from commaqa.inference.model_search import ParticipantModel

logger = logging.getLogger(__name__)


class ExecutionParticipant(ParticipantModel):
    def __init__(self, remodel_file=None, next_model="gen", skip_empty_answers=False):
        self.next_model = next_model
        self.skip_empty_answers = skip_empty_answers
        self.per_model_calls = {"executer": 0, "op_executer": 0}
        self.kb_lang_groups = None
        self.model_lib = None
        self.operation_regex = re.compile("\((.+)\) \[([^\]]+)\] (.*)")
        self.remodel_mode = False
        if remodel_file:
            self.remodel_mode = True
            with open(remodel_file, "r") as input_fp:
                input_json = json.load(input_fp)
            self.kb_lang_groups = []
            self.qid_to_kb_lang_idx = {}
            for input_item in input_json:
                kb = input_item["kb"]
                pred_lang = input_item["pred_lang_config"]
                idx = len(self.kb_lang_groups)
                self.kb_lang_groups.append((kb, pred_lang))
                for qa_pair in input_item["qa_pairs"]:
                    qid = qa_pair["id"]
                    self.qid_to_kb_lang_idx[qid] = idx

    def set_model_lib(self, model_map):
        # ignore any remodel file input once this is set
        self.remodel_mode = False
        self.model_lib = model_map

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
        self.per_model_calls["executer"] += 1
        step_model_key = "executer_step{}".format(len(data.get_current_qseq()))
        if step_model_key not in self.per_model_calls:
            self.per_model_calls[step_model_key] = 0
        self.per_model_calls[step_model_key] += 1

        question = data.get_last_question()
        qid = data["qid"]
        if self.remodel_mode:
            (kb, pred_lang) = self.kb_lang_groups[self.qid_to_kb_lang_idx[qid]]
            model_configurations = {}
            for model_name, configs in pred_lang.items():
                model_configurations[model_name] = [ModelQuestionConfig(config) for config in configs]
            self.model_lib = build_models(model_configurations, kb, ignore_input_mismatch=True)
            context = None
        else:
            if "paras" not in data:
                print(data.keys())
                raise ValueError(
                    "Make sure to set add_paras to True in dataset reader config" " to use the context paras"
                )
            context = data["paras"]
        ### run the model (as before)
        if debug:
            print("<OPERATION>: %s, qid=%s" % (question, qid))
        m = self.operation_regex.match(question)
        if m is None:
            logger.debug("No match for {}".format(question))
            return []
        assignment = {}
        for ans_idx, ans in enumerate(data.get_current_aseq()):
            assignment["#" + str(ans_idx + 1)] = json.loads(ans)
        executer = OperationExecuter(model_library=self.model_lib, ignore_input_mismatch=True)
        answers, facts_used = executer.execute_operation(
            operation=m.group(1), model=m.group(2), question=m.group(3), assignments=assignment, context=context
        )
        if self.remodel_mode:
            for model_name, model in self.model_lib.items():
                if model_name not in self.per_model_calls:
                    self.per_model_calls[model_name] = 0
                self.per_model_calls[model_name] += model.num_calls

        self.per_model_calls["op_executer"] += executer.num_calls
        # reset calls if executer reused
        executer.num_calls = 0

        if not valid_answer(answers):
            logger.debug(
                "Invalid answer for qid: {} question: {} chain: {}!".format(
                    qid, question, ", ".join(data.get_current_qseq())
                )
            )
            return []
        if self.skip_empty_answers and not nonempty_answer(answers):
            logger.debug(
                "Empty answer for qid: {} question: {} chain: {}!".format(
                    qid, question, ", ".join(data.get_current_qseq)
                )
            )
            return []

        # copy state
        new_state = state.copy()

        ## add answer
        new_state.data.add_answer(
            QuestionAnsweringStep(
                answer=json.dumps(answers),
                model=m.group(2),
                subquestion=m.group(3),
                operation=m.group(1),
                score=0,
                participant="execution",
            )
        )
        new_state.next = self.next_model

        return [new_state]
