import torch
import torch.nn as nn
from typing import Optional
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Decoder(nn.Module):
    """
    Represents a single decoder block within a transformer-based model, which processes input sequences with attention
    to the encoder's output and its own previous outputs. It employs masked multi-head self-attention to maintain
    auto-regressive properties for tasks like language modeling.

    Attributes:
        masked_multihead_attention (MultiHeadAttention): A masked multi-head self-attention mechanism that allows
            the decoder to focus on different parts of the decoder's input sequence without attending to future tokens.
        multihead_attention (MultiHeadAttention): A multi-head attention mechanism that attends to the encoder's output,
            integrating context from the encoder into the decoder's process.
        feed_forward (FeedForward): A position-wise feed-forward neural network applied after the attention mechanisms.
        norm1, norm2, norm3 (nn.LayerNorm): Layer normalization layers applied before and after the self-attention and
            encoder-decoder attention mechanisms, and before the feed-forward network, respectively.
        dropout1, dropout2, dropout3, dropout_ffn (nn.Dropout): Dropout layers applied after each attention mechanism
            and the feed-forward network to reduce overfitting.

    Args:
        config (object): A configuration object containing hyperparameters for initializing the decoder block components,
            including the hidden layer size (`hidden_size`), dropout probability (`hidden_dropout_prob`), and other
            parameters required by the `MultiHeadAttention` and `FeedForward` modules.
    """
    def __init__(self, config: object) -> None:
        super(Decoder, self).__init__()
        self.masked_multihead_attention = MultiHeadAttention(config)
        self.multihead_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_ffn = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, 
        hidden_state: torch.Tensor, 
        encoder_info: torch.Tensor, 
        biases: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes the input hidden states through a decoder block, applying masked self-attention, encoder-decoder
        attention, a feed-forward network, layer normalization, and dropout.

        Args:
            hidden_state (torch.Tensor): The input tensor containing hidden states for each token in the decoder's input sequence.
            encoder_info (torch.Tensor): The output tensor from the encoder, providing context for each token in the encoder's input sequence.
            biases (torch.Tensor): Bias tensor used in the self-attention mechanism to prevent attention to certain positions.
            mask (Optional[torch.Tensor]): Optional attention mask tensor applied during self-attention, allowing the model to ignore specific tokens for attention calculations (e.g., padding tokens).

        Returns:
            torch.Tensor: The output tensor after processing through the decoder block, maintaining the same shape as the input hidden_state tensor.
        """
        input_shape = hidden_state.size()
        causal_mask = self.get_causal_attention_mask(input_shape)

        if mask is not None:
            padding_mask = mask.unsqueeze(1).expand_as(causal_mask)
            causal_mask = torch.min(padding_mask, causal_mask)

        normed_hidden_state = self.norm1(hidden_state)
        attention_output = self.masked_multihead_attention(normed_hidden_state, normed_hidden_state, normed_hidden_state, biases, causal_mask)
        attention_output = self.dropout1(attention_output)
        hidden_state += attention_output

        normed_hidden_state = self.norm2(hidden_state)
        attention_output = self.multihead_attention(normed_hidden_state, encoder_info, encoder_info, biases, mask)
        attention_output = self.dropout2(attention_output)
        hidden_state += attention_output

        normed_hidden_state = self.norm3(hidden_state)
        feed_forward_output = self.feed_forward(normed_hidden_state)
        feed_forward_output = self.dropout3(feed_forward_output)
        hidden_state += feed_forward_output

        hidden_state = self.dropout_ffn(hidden_state)

        return hidden_state

    def get_causal_attention_mask(self, input_shape: torch.Size) -> torch.Tensor:
        """
        Generates a causal attention mask to prevent decoder tokens from attending to future tokens in the sequence.

        Args:
            input_shape (torch.Size): The shape of the input tensor, used to determine the batch size and sequence length for generating the mask.

        Returns:
            torch.Tensor: A causal attention mask tensor of shape [batch_size, sequence_length, sequence_length], ensuring auto-regressive properties in the decoder.
        """
        batch_size, sequence_length = input_shape[0], input_shape[1]
        device: torch.device = next(self.parameters()).device
        mask: torch.Tensor = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.float, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).expand(batch_size, -1, -1)
