import torch
import torch.nn as nn
from relative_position_bias import RelativePositionBias

class Embeddings(nn.Module):
    """
    The Embeddings class is a crucial component of transformer-based models, responsible for converting input token IDs into dense vector representations. These embeddings are the first step in processing input data, transforming discrete token IDs into continuous vectors that can encapsulate semantic information. This class not only performs token embedding lookup but also applies layer normalization and dropout for regularization.

    The embedding vectors produced by this class serve as the input to subsequent layers of the model, facilitating the learning of token-specific features in the context of the task at hand, whether it be language understanding, translation, or another natural language processing (NLP) application.

    Attributes:
        hidden_size (int): The size of the embedding vectors. This is also the size of the hidden layers throughout the model.
        vocab_size (int): The size of the vocabulary, determining the number of unique token embeddings the layer can produce.
        hidden_dropout_prob (float): The dropout probability, used to randomly zero elements of the embedding vectors with this probability to prevent overfitting.
        token_embeddings (nn.Embedding): The PyTorch embedding layer that maps token IDs to embedding vectors.
        dropout (nn.Dropout): The dropout layer applied to the embedding vectors.
        norm (nn.LayerNorm): The layer normalization applied to the embedding vectors.

    Methods:
        forward(input_ids: torch.Tensor) -> torch.Tensor
            Performs the embedding lookup, followed by layer normalization and dropout, to produce the final embedding vectors for the input tokens.

    Args:
        config (object): A configuration object containing the attributes necessary to initialize the embedding layer. These attributes include:
            - hidden_size: The dimensionality of the embedding vectors and the model's hidden layers.
            - vocab_size: The total size of the vocabulary.
            - hidden_dropout_prob: The dropout probability for regularization.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hidden_dropout_prob = config.hidden_dropout_prob

        # Initialize the embedding layer, dropout, and layer normalization
        self.token_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Processes input token IDs through the embedding layer, followed by layer normalization and dropout.

        Args:
            input_ids (torch.Tensor): A tensor containing a batch of input token IDs.

        Returns:
            torch.Tensor: The resulting tensor after applying embeddings, layer normalization, and dropout. This tensor is ready to be fed into subsequent layers of the model.
        """
        x = self.token_embeddings(input_ids)  # Embedding lookup
        x = self.norm(x)  # Apply layer normalization
        x = self.dropout(x)  # Apply dropout
        return x

