import torch

from allennlp.nn.util import replace_masked_values, min_value_of_dtype, get_range_vector, get_device_of


def get_best_span(
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        max_length: int = None
    ) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_mask = torch.triu(torch.ones((passage_length, passage_length), device=device))

    if max_length is not None:
        range_vector = get_range_vector(passage_length, get_device_of(span_start_logits))
        range_matrix = range_vector.unsqueeze(0)-range_vector.unsqueeze(1)
        length_mask = ((range_matrix < max_length) & (range_matrix >= 0))
        span_mask = (span_mask.long() & length_mask).float()

    span_log_mask = span_mask.log()

    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)


def get_best_k_spans(
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        num_spans: int,
        max_length: int = None
    ) -> torch.Tensor:

    best_spans = []
    range_vector = get_range_vector(span_start_logits.shape[0], get_device_of(span_start_logits))
    for _ in range(num_spans):
        best_span = get_best_span(span_start_logits, span_end_logits, max_length)

        mask = torch.ones_like(span_start_logits, dtype=torch.bool)

        # Option 1
        mask[range_vector, best_span[:, 0]] = False
        mask[range_vector, best_span[:, 1]] = False

        # # Option 2 (seems too extreme)
        # for i, (start, end) in enumerate(best_span):
        #     mask[i, start : end + 1] = False

        span_start_logits = replace_masked_values_with_big_negative_number(span_start_logits, mask)
        span_end_logits = replace_masked_values_with_big_negative_number(span_end_logits, mask)

        best_spans.append(best_span)

    best_spans = torch.stack(best_spans, dim=1)
    return best_spans


def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    Replace the masked values in a tensor something really negative so that they won't
    affect a max operation.
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))
