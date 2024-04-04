import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from embeddings import Embeddings
from relative_position_bias import RelativePositionBias

class T5Model(nn.Module):
    """
    Implements a simplified version of the T5 model, a transformer-based model designed for a variety of NLP tasks. This model consists of an embedding layer, multiple encoder and decoder blocks, and a final prediction layer. The model is designed to process sequences of tokens, encoding them into a latent space representation which is then decoded into an output sequence.

    The architecture is modular, allowing for a configurable number of encoder and decoder blocks. Each block in the encoder and decoder is capable of self-attention and feed-forward neural network processing. The model also incorporates relative position biases to account for the positions of tokens within the sequence.

    Attributes:
        num_blocks (int): The number of encoder and decoder blocks to include in the model.
        vocab_size (int): The size of the vocabulary used in the embeddings layer.
        hidden_size (int): The dimensionality of the hidden layers and embeddings.
        embed_layer (Embeddings): The initial embedding layer for input tokens.
        relative_position_bias (RelativePositionBias): Module for calculating relative position biases used in attention mechanisms.
        biases (torch.Tensor): Pre-computed biases for all positions up to a maximum token length.
        encoder (nn.ModuleList): A list of encoder blocks.
        decoder (nn.ModuleList): A list of decoder blocks.
        prediction_layer (nn.Linear): A linear layer that projects decoder output to the vocabulary size.
        softmax (nn.LogSoftmax): The softmax layer applied to the outputs of the prediction layer.

    Methods:
        forward(input_ids, labels, mask) -> Tuple[torch.Tensor, torch.Tensor]
            Processes input token IDs and labels through the model, returning logits for the predicted token IDs.
        to(*args, **kwargs)
            Overrides the `.to()` method to ensure all parts of the model, including manually managed tensors, are moved to the specified device.

    Args:
        config (object): A configuration object with hyperparameters for model components. Expected attributes include `num_blocks`, `vocab_size`, `hidden_size`, among others necessary for initializing submodules.
    """
    def __init__(self, config):
        super(T5Model, self).__init__()

        self.num_blocks: int = config.num_blocks
        self.vocab_size: int = config.vocab_size
        self.hidden_size: int = config.hidden_size

        self.embed_layer: Embeddings = Embeddings(config)
        self.relative_position_bias = RelativePositionBias(config)
        self.biases = self.relative_position_bias(config.max_token_len, config.max_token_len).to(self.embed_layer.token_embeddings.weight.device)

        self.encoder: nn.ModuleList = nn.ModuleList([Encoder(config) for _ in range(self.num_blocks)])
        self.decoder: nn.ModuleList = nn.ModuleList([Decoder(config) for _ in range(self.num_blocks)])

        # The output of the final decoder block is fed into a dense layer with a softmax output, whose weights are shared with the input embedding matrix.
        self.prediction_layer: nn.Linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.prediction_layer.weight = self.embed_layer.token_embeddings.weight
        self.softmax: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes input and label sequences through the T5 model architecture, returning the logits of the predicted output sequence.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            labels (torch.Tensor): Tensor of target token IDs for teacher forcing during training.
            mask (torch.Tensor): Attention mask tensor to specify which positions should be attended to and which should not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The logits representing the model's predictions for the next token in the sequence.
        """
        print(self.embed_layer.token_embeddings.weight.device)
        print(self.biases.device)
        x_enc: torch.Tensor  = self.embed_layer(input_ids)
        x_dec: torch.Tensor  = self.embed_layer(labels)

        for encoder_layer in self.encoder:
            x_enc: torch.Tensor = encoder_layer(x_enc, self.biases, mask)

        for decoder_layer in self.decoder:
            x_dec = decoder_layer(x_dec, x_enc, self.biases, mask)

        x_logits: torch.Tensor = self.prediction_layer(x_dec)

        return self.softmax(x_logits)

    def to(self, *args, **kwargs):
        """
        Ensures the model and its components are moved to the specified device. This method is particularly necessary for moving tensors that are not automatically managed by PyTorch's `.to()` method.

        Args:
            *args: Positional arguments for the device specification.
            **kwargs: Keyword arguments for the device specification.
        """
        super(T5Model, self).to(*args, **kwargs)
        self.biases = self.biases.to(*args, **kwargs)
        self.relative_position_bias.to(*args, **kwargs)
        
        return self


