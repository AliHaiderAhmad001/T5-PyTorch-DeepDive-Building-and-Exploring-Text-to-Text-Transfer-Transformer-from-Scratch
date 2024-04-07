import torch
import torch.nn as nn
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.hidden_dropout_prob: float = config.hidden_dropout_prob

        self.multihead_attention: MultiHeadAttention = MultiHeadAttention(config)
        self.feed_forward: FeedForward = FeedForward(config)

        self.norm1: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.norm2: nn.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout1: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)
        self.dropout2: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)
        self.dropout_ffn: nn.Dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, 
                hidden_state: torch.Tensor, 
                biases: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Processes input hidden states through an encoder block, applying self-attention, 
        feed-forward network, normalization, and dropout.

        Args:
            hidden_state (torch.Tensor): The input tensor containing hidden states for each token in the input sequence.
            biases (torch.Tensor): The bias tensor used in the self-attention mechanism to prevent attention to certain positions.
            mask (Optional[torch.Tensor]): An optional mask tensor to apply during the self-attention mechanism, 
                                            allowing the model to ignore specific tokens for attention calculations 
                                            (e.g., padding tokens).

        Returns:
            torch.Tensor: The output tensor after processing through the encoder block, 
                          with the same shape as the input hidden_state tensor.
        """
        normed_hidden_state: torch.Tensor = self.norm1(hidden_state)
        attention_output: torch.Tensor = self.multihead_attention(
            query=normed_hidden_state, 
            key=normed_hidden_state, 
            value=normed_hidden_state, 
            relative_position_bias=biases, 
            mask=mask
        )
        attention_output: torch.Tensor = self.dropout1(attention_output)
        hidden_state: torch.Tensor = hidden_state + attention_output

        normed_hidden_state: torch.Tensor = self.norm2(hidden_state)
        feed_forward_output: torch.Tensor = self.feed_forward(normed_hidden_state)
        feed_forward_output: torch.Tensor = self.dropout2(feed_forward_output)
        hidden_state: torch.Tensor = hidden_state + feed_forward_output

        hidden_state: torch.Tensor = self.dropout_ffn(hidden_state)

        return hidden_state
