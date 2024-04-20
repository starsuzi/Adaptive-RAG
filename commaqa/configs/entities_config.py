import random
from math import ceil
from typing import Dict, List


class EntitiesConfig:
    def __init__(self, entities_json: Dict[str, List[str]]):
        self.entity_type_map = entities_json

    def subsample(self, num_ents):
        new_ent_map = {}
        for etype, elist in self.entity_type_map.items():
            # if fraction passed, sample ratio
            if num_ents <= 1:
                new_ent_map[etype] = random.sample(elist, ceil(len(elist) * num_ents))
            else:
                new_ent_map[etype] = random.sample(elist, num_ents)

        return EntitiesConfig(new_ent_map)

    def __getitem__(self, item: str):
        return self.entity_type_map[item]
