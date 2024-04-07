import torch
import torch.nn as nn
from relation_aware_attention import MultiHeadAttention
from feed_forward import FeedForward

class Encoder(nn.Module):
    """
    The Encoder class represents a single encoder block within a transformer-based model, encapsulating the core components of self-attention and feed-forward neural network layers. This block is designed to process input sequences through multi-head self-attention, followed by position-wise feed-forward neural network layers, incorporating normalization and dropout at various stages to enhance training stability and prevent overfitting.

    Attributes:
        hidden_size (int): The size of the hidden layers and embeddings in the transformer model.
        hidden_dropout_prob (float): The probability of dropout applied after the multi-head attention and feed-forward network layers to prevent overfitting.
        multihead_attention (MultiHeadAttention): The multi-head self-attention mechanism allowing the model to focus on different parts of the input sequence.
        feed_forward (FeedForward): The position-wise feed-forward neural network applied after the attention mechanism.
        norm1 (nn.LayerNorm): The layer normalization applied before the multi-head attention mechanism.
        norm2 (nn.LayerNorm): The layer normalization applied before the feed-forward network.
        dropout1 (nn.Dropout): The dropout applied after the multi-head attention mechanism.
        dropout2 (nn.Dropout): The dropout applied after the feed-forward network.
        dropout_ffn (nn.Dropout): An additional dropout layer applied after the final addition of the feed-forward network output.

    Methods:
        forward(hidden_state, biases, mask=None) -> torch.Tensor
            Performs a forward pass through the encoder block, processing the input hidden states with self-attention and feed-forward layers.

    Args:
        config (object): A configuration object containing hyperparameters for initializing the encoder block components. These include the hidden layer size (`hidden_size`), dropout probability (`hidden_dropout_prob`), and other parameters required by the `MultiHeadAttention` and `FeedForward` modules.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob

        # Initialize the multi-head attention and feed-forward components
        self.multihead_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        # Layer normalization and dropout for stabilization and regularization
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout1 = nn.Dropout(self.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(self.hidden_dropout_prob)
        self.dropout_ffn = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state, biases, mask=None):
        """
        Processes input hidden states through an encoder block, applying self-attention, feed-forward network, normalization, and dropout.

        Args:
            hidden_state (torch.Tensor): The input tensor containing hidden states for each token in the input sequence.
            biases (torch.Tensor): The bias tensor used in the self-attention mechanism to prevent attention to certain positions.
            mask (torch.Tensor, optional): An optional mask tensor to apply during the self-attention mechanism, allowing the model to ignore specific tokens for attention calculations (e.g., padding tokens).

        Returns:
            torch.Tensor: The output tensor after processing through the encoder block, with the same shape as the input hidden_state tensor.
        """
        # Apply layer normalization, self-attention, and dropout
        normed_hidden_state = self.norm1(hidden_state)
        attention_output = self.multihead_attention(normed_hidden_state, normed_hidden_state, normed_hidden_state, biases, mask)
        attention_output = self.dropout1(attention_output)
        hidden_state = hidden_state + attention_output

        # Apply layer normalization, feed-forward network, and dropout
        normed_hidden_state = self.norm2(hidden_state)
        feed_forward_output = self.feed_forward(normed_hidden_state)
        feed_forward_output = self.dropout2(feed_forward_output)
        hidden_state = hidden_state + feed_forward_output

        # Apply an additional dropout layer
        hidden_state = self.dropout_ffn(hidden_state)

        return hidden_state

