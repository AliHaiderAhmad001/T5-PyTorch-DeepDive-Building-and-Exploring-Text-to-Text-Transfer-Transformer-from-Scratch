import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """
    Relation-aware attention head implementation.

    Args:
        hidden_size (int): Hidden size for the model (embedding dimension).
        head_dim (int): Dimensionality of the attention head.

    Attributes:
        query_weights (nn.Linear): Linear layer for query projection.
        key_weights (nn.Linear): Linear layer for key projection.
        value_weights (nn.Linear): Linear layer for value projection.
    """

    def __init__(self, hidden_size, head_dim, hidden_dropout_prob=0.1):
        """
        Initializes the AttentionHead.

        Args:
            hidden_size (int): Hidden size for the model (embedding dimension).
            head_dim (int): Dimensionality of the attention head.
            hidden_dropout_prob(float)
        """
        super().__init__()
        self.head_dim = head_dim
        self.query_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, head_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 relative_biases:torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies attention mechanism to the input query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Optional mask tensor.

        Returns:
            torch.Tensor: Updated value embeddings after applying attention mechanism.
        """
        query: torch.Tensor = self.query_weights(query)
        key: torch.Tensor = self.key_weights(key)
        value: torch.Tensor = self.value_weights(value)

        att_scores: torch.Tensor = (torch.matmul(query, key.transpose(1, 2)) + relative_biases) / self.head_dim ** 0.5
        
        if mask is not None:
            if mask.dim() == 2:
                # Padding mask case: [batch_size, seq_len]
                # Unsqueeze to [batch_size, 1, seq_len] to match the broadcasting requirements
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3:
                # Already in [batch_size, seq_len, seq_len] shape, no adjustment needed
                pass
            else:
                raise ValueError("Mask dimension is not supported. Must be 2 or 3.")

            # Apply mask - inf where mask == 0, keep original scores where mask != 0
            att_scores = att_scores.masked_fill(mask == 0, float('-inf'))


        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)
        att_weights: torch.Tensor = self.dropout(att_weights)
        n_value: torch.Tensor = torch.matmul(att_weights, value)

        return n_value
        
        
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer implementation.
    """

    def __init__(self, config):
        """
        Initializes the MultiHeadAttention layer.
        """
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_heads
        self.head_dim: int = self.hidden_size // self.num_heads
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_heads: nn.ModuleList = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim, self.hidden_dropout_prob) for head_num in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, relative_position_bias: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies multi-head attention mechanism to the input query, key, and value tensors.
        """
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask, relative_biases=relative_position_bias[:,i]) for i, attention_head in enumerate(self.attention_heads)]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state
