from typing import Tuple

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("full_recall_rank")
class FullRecallRank(Metric):
    """
    ListCompareEmAndF1: Em and F1 (Similar to HotpotQA Sp metric)
    """
    def __init__(self) -> None:
        self._total_rank = 0.0
        self._count = 0

    def __call__(self,
                 prediction: torch.Tensor,
                 gold_label: torch.Tensor):
        # both are tensors of shape 1 and are lists of indices.
        prediction, gold_label = self.detach_tensors(prediction, gold_label)

        predicted_ranked_ids = list(map(int, prediction)) # ordered / ranked list
        gold_ids = set(map(int, gold_label))

        if not gold_ids:
            # Temporary hack. The total_recall_rank isn't true in this case.
            self._total_rank += 100
        else:
            current_rank = max(predicted_ranked_ids.index(gold_id) for gold_id in gold_ids) + 1
            self._total_rank += current_rank
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        average_rank = self._total_rank / self._count if self._count > 0 else 0

        if reset:
            self.reset()
        return average_rank

    @overrides
    def reset(self):
        self._total_rank = 0.0
        self._count = 0.0
