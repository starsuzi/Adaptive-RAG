from typing import Tuple
import torch

def pluck_tokens(paragraph_tensor: torch.Tensor,
                 token_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
    paragraph_tensor: (batch_size, paragraph_max_seq_len, ...)
    token_positions: (batch_size, max_num_tokens)
    Output:
    plucked_tokens_tensor: (batch_size, max_num_tokens, ...)
    """

    instance_plucked_tokens_tensors = []
    token_positions_mask = (token_positions != -1)
    token_positions[~token_positions_mask] = 0

    for instance_paragraph_tensor, instance_token_positions in zip(torch.unbind(paragraph_tensor, dim=0),
                                                                   torch.unbind(token_positions, dim=0)):
        plucked_tokens_tensor = torch.index_select(instance_paragraph_tensor, 0, instance_token_positions)
        instance_plucked_tokens_tensors.append(plucked_tokens_tensor)

    plucked_tokens_tensor = torch.nn.utils.rnn.pad_sequence(instance_plucked_tokens_tensors, batch_first=True)
    plucked_tokens_tensor = plucked_tokens_tensor*token_positions_mask.unsqueeze(-1).float()
    return plucked_tokens_tensor, token_positions_mask
