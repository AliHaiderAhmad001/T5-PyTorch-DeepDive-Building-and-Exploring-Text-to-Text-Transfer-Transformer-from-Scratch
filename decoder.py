import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Decoder(nn.Module):
    """
    The Decoder class represents a single decoder block within a transformer-based model. This block is designed to process input sequences with attention to both the output of the encoder and its own previous outputs. The decoder employs masked multi-head self-attention to prevent positions from attending to subsequent positions. This is crucial for preserving the auto-regressive property in tasks like language modeling.

    Attributes:
        masked_multihead_attention (MultiHeadAttention): The masked multi-head self-attention mechanism that allows the decoder to focus on different parts of the decoder's input sequence without looking ahead to future tokens.
        multihead_attention (MultiHeadAttention): The multi-head attention mechanism that focuses on the encoder's output, facilitating the integration of context from the encoder.
        feed_forward (FeedForward): The position-wise feed-forward neural network applied after the attention mechanisms.
        norm1, norm2, norm3 (nn.LayerNorm): Layer normalization applied before and after the self-attention and encoder-decoder attention mechanisms, as well as before the feed-forward network.
        dropout1, dropout2, dropout3, dropout_ffn (nn.Dropout): Dropout layers applied after each attention mechanism and the feed-forward network to prevent overfitting.

    Methods:
        forward(hidden_state, encoder_info, biases, mask=None) -> torch.Tensor
            Performs a forward pass through the decoder block, processing the input hidden states with masked self-attention, encoder-decoder attention, and feed-forward layers.

        get_causal_attention_mask(input_shape) -> torch.Tensor
            Generates a causal attention mask to prevent decoder tokens from attending to future tokens in the sequence.

    Args:
        config (object): A configuration object containing hyperparameters for initializing the decoder block components. These include the hidden layer size (`hidden_size`), dropout probability (`hidden_dropout_prob`), and other parameters required by the `MultiHeadAttention` and `FeedForward` modules.
    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        # Initialize the attention mechanisms and feed-forward network
        self.masked_multihead_attention = MultiHeadAttention(config)
        self.multihead_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        # Layer normalization and dropout for stabilization and regularization
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout3 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_ffn = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state, encoder_info, biases, mask=None):
        """
        Processes input hidden states through a decoder block.

        Args:
            hidden_state (torch.Tensor): The input tensor containing hidden states for each token in the decoder's input sequence.
            encoder_info (torch.Tensor): The output tensor from the encoder, containing context for each token in the encoder's input sequence.
            biases (torch.Tensor): The bias tensor used in the self-attention mechanism to prevent attention to certain positions.
            mask (torch.Tensor, optional): An optional mask tensor to apply during the self-attention mechanism, allowing the model to ignore specific tokens for attention calculations (e.g., padding tokens).

        Returns:
            torch.Tensor: The output tensor after processing through the decoder block, with the same shape as the input hidden_state tensor.
        """

        input_shape = hidden_state.size()
        causal_mask = self.get_causal_attention_mask(input_shape)

        if mask is not None:
            padding_mask = mask.unsqueeze(1).expand_as(causal_mask)
            causal_mask = torch.min(padding_mask, causal_mask)

        normed_hidden_state = self.norm1(hidden_state)
        attention_output = self.masked_multihead_attention(normed_hidden_state, normed_hidden_state, normed_hidden_state, biases, causal_mask)
        attention_output = self.dropout1(attention_output)
        hidden_state = attention_output + hidden_state
        #print(attention_output.shape)
        normed_hidden_state = self.norm2(hidden_state)
        attention_output = self.multihead_attention(normed_hidden_state, encoder_info, encoder_info, biases, padding_mask)

        attention_output = self.dropout2(attention_output)
        hidden_state = attention_output + hidden_state

        normed_hidden_state = self.norm3(hidden_state)
        feed_forward_output = self.feed_forward(normed_hidden_state)
        feed_forward_output = self.dropout3(feed_forward_output)
        hidden_state = feed_forward_output + hidden_state

        hidden_state = self.dropout_ffn(hidden_state)

        return hidden_state

    def get_causal_attention_mask(self, input_shape):
        """
        Generates a causal attention mask to ensure decoder self-attention is auto-regressive.

        Args:
            input_shape: A tuple representing the shape of inputs to the decoder.

        Returns:
            torch.Tensor: A causal attention mask tensor used to mask future tokens in the sequence during self-attention.
        """
        batch_size, sequence_length = input_shape[0], input_shape[1]
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones((sequence_length, sequence_length), dtype=torch.float32, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1) 
        return mask


