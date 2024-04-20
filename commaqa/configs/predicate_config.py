import random
from copy import deepcopy
from typing import List, Dict

from commaqa.configs.entities_config import EntitiesConfig
from commaqa.dataset.utils import get_predicate_args


class PredicateConfig:
    def __init__(self, pred_json):
        self.pred_name = pred_json[0]
        self.args = pred_json[1]["args"]
        self.nary = pred_json[1].get("nary")
        self.graph_type = pred_json[1].get("type")
        self.language = pred_json[1].get("language")

    def populate_chains(self, entity_config: EntitiesConfig) -> List[str]:
        if len(self.args) != 2 or self.args[0] != self.args[1]:
            raise ValueError(
                "Chains KB can only be created with binary predicates having the same "
                "arg types. Change args for {}".format(self.pred_name)
            )
        kb = []
        entity_list = deepcopy(entity_config.entity_type_map[self.args[0]])
        last_entity = None
        while len(entity_list) > 0:
            if last_entity is None:
                last_entity = random.choice(entity_list)
                entity_list.remove(last_entity)
            next_entity = random.choice(entity_list)
            entity_arr = [last_entity, next_entity]
            fact = self.pred_name + "(" + ", ".join(entity_arr) + ")"
            kb.append(fact)
            last_entity = next_entity
            entity_list.remove(last_entity)
        return kb

    def populate_trees(self, entity_config: EntitiesConfig) -> List[str]:
        if len(self.args) != 2 or self.args[0] != self.args[1]:
            raise ValueError(
                "Trees KB can only be created with binary predicates having the same "
                "arg types. Change args for {}".format(self.pred_name)
            )
        if len(self.nary) is None or "1" not in self.nary:
            raise ValueError(
                "Nary needs to be set with at least one index set to 1 to produce"
                "a tree structure kb. Pred: {}".format(self.pred_name)
            )
        kb = []
        entity_list = deepcopy(entity_config.entity_type_map[self.args[0]])
        # create the root node
        open_entities = random.sample(entity_list, 1)
        entity_list.remove(open_entities[0])
        unique_idx = self.nary.index("1")
        while len(entity_list) > 0:
            new_open_entities = []
            # for each open node
            for open_entity in open_entities:
                # select children
                if len(entity_list) > 2:
                    children = random.sample(entity_list, 2)
                else:
                    children = entity_list
                # add edge between child and open node
                for child in children:
                    if unique_idx == 1:
                        entity_arr = [open_entity, child]
                    else:
                        entity_arr = [child, open_entity]
                    # remove child from valid nodes to add
                    entity_list.remove(child)
                    # add it to the next set of open nodes
                    new_open_entities.append(child)
                fact = self.pred_name + "(" + ", ".join(entity_arr) + ")"
                kb.append(fact)
            open_entities = new_open_entities
        return kb

    def populate_kb(self, entity_config: EntitiesConfig) -> List[str]:
        if self.graph_type == "chain":
            return self.populate_chains(entity_config)
        elif self.graph_type == "tree":
            return self.populate_tree(entity_config)
        elif self.graph_type is not None:
            raise ValueError("Unknown graph type: {}".format(self.graph_type))
        if self.nary is None:
            raise ValueError("At least one of nary or type needs to be set for predicate" " {}".format(self.pred_name))

        return self.populate_relations(entity_config)

    def populate_relations(self, entity_config: EntitiesConfig) -> List[str]:
        kb = set()
        arg_counts = []
        arg_pos_list = []
        for arg in self.args:
            if arg not in entity_config.entity_type_map:
                raise ValueError(
                    "No entity list defined for {}." "Needed for predicate: {}".format(arg, self.pred_name)
                )
            arg_counts.append(len(entity_config.entity_type_map[arg]))
            arg_pos_list.append(deepcopy(entity_config.entity_type_map[arg]))

        max_attempts = 2 * max(arg_counts)
        orig_arg_pos_list = deepcopy(arg_pos_list)
        while max_attempts > 0:
            entity_arr = []
            max_attempts -= 1
            for idx in range(len(self.args)):
                ent = random.choice(arg_pos_list[idx])
                # assume relations can never be reflexive
                if ent in entity_arr:
                    entity_arr = None
                    break
                entity_arr.append(ent)
            if entity_arr is None:
                continue
            for idx, ent in enumerate(entity_arr):
                if self.nary[idx] == "1":
                    arg_pos_list[idx].remove(ent)
                    if len(arg_pos_list[idx]) == 0:
                        max_attempts = 0
                elif self.nary[idx] == "n":
                    arg_pos_list[idx].remove(ent)
                    # once all entities have been used once, reset to the original list
                    if len(arg_pos_list[idx]) == 0:
                        arg_pos_list[idx] = deepcopy(orig_arg_pos_list[idx])
            fact = self.pred_name + "(" + ", ".join(entity_arr) + ")"
            if fact not in kb:
                kb.add(fact)
        return list(kb)

    def generate_kb_fact_map(self, kb: Dict[str, List[str]]) -> Dict[str, str]:
        kb_fact_map = {}
        for kb_item in kb[self.pred_name]:
            if self.language:
                pred, args = get_predicate_args(kb_item)
                sentence = self.language if isinstance(self.language, str) else random.choice(self.language)
                for argidx, arg in enumerate(args):
                    sentence = sentence.replace("$" + str(argidx + 1), arg)
            else:
                pred_name, fields = get_predicate_args(kb_item)
                if len(fields) != 2:
                    sentence = kb_item
                else:
                    sentence = fields[0] + " " + pred_name + " " + " ".join(fields[1:])
            kb_fact_map[kb_item] = sentence + "."
        return kb_fact_map

    def generate_context(self, kb: Dict[str, List[str]]) -> str:
        kb_fact_map = self.generate_kb_fact_map(kb)
        return " ".join(kb_fact_map.values())
