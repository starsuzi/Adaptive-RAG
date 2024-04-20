import logging

from commaqa.dataset.utils import get_predicate_args

logger = logging.getLogger(__name__)


class KBLookup:
    def __init__(self, kb):
        self.kb = kb

    def ask_question(self, question_predicate, context=None):
        if context:
            raise ValueError("Input context passed to KBLookup which does not use context!" + "\n{}".format(context))
        return self.ask_question_predicate(question_predicate)

    def ask_question_predicate(self, question_predicate):
        predicate, pred_args = get_predicate_args(question_predicate)
        answers = []
        facts_used = []
        for fact in self.kb[predicate]:
            fact_pred, fact_args = get_predicate_args(fact)
            if len(pred_args) != len(fact_args):
                raise ValueError("Mismatch in specification args {} and fact args {}".format(pred_args, fact_args))
            mismatch = False
            answer = ""
            for p, f in zip(pred_args, fact_args):
                # KB fact arg doesn't match the predicate arg
                if p != "?" and p != f and p != "_":
                    mismatch = True
                # predicate arg is a query, populate answer with fact arg
                elif p == "?":
                    answer = f
            # if all args matched, add answer
            if not mismatch:
                answers.append(answer)
                facts_used.append(fact)
        if len(answers) == 0:
            logger.debug("No matching facts for {}. Facts:\n{}".format(question_predicate, self.kb[predicate]))

        # If its a boolean query, use number of answers
        if "?" not in pred_args:
            if len(answers) == 0:
                return "no", facts_used
            else:
                return "yes", facts_used
        else:
            return answers, facts_used
