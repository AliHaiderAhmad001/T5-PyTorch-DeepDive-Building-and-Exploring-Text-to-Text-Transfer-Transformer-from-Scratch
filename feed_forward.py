class FeedForward(nn.Module):
    """
    The FeedForward class represents the feed-forward neural network component commonly found in transformer-based models. This component consists of two linear layers with a GELU activation function and dropout applied in between.

    Attributes:
        hidden_size (int): The dimensionality of the input and output tensors.
        intermediate_fc_size (int): The size of the intermediate fully connected layer, typically set to four times the hidden size.
        hidden_dropout_prob (float): The dropout probability applied after the GELU activation function.

    Methods:
        forward(hidden_state: torch.Tensor) -> torch.Tensor
            Performs the forward pass through the feed-forward network, applying linear transformations, activation function, and dropout.

    Args:
        config (object): A configuration object containing the attributes necessary to initialize the feed-forward network. These attributes include:
            - hidden_size: The dimensionality of the input and output tensors.
            - hidden_dropout_prob: The dropout probability applied after the GELU activation function.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_fc_size = self.hidden_size * 4
        self.hidden_dropout_prob = config.hidden_dropout_prob

        # Define the linear layers and dropout
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_fc_size)
        self.fc2 = nn.Linear(self.intermediate_fc_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the feed-forward network.

        Args:
            hidden_state (torch.Tensor): The input tensor representing hidden states from the previous layer.

        Returns:
            torch.Tensor: The output tensor after passing through the feed-forward network.
        """
        hidden_state = self.fc1(hidden_state)  # Linear transformation
        hidden_state = F.gelu(hidden_state)  # Apply GELU activation function
        hidden_state = self.dropout(hidden_state)  # Apply dropout
        hidden_state = self.fc2(hidden_state)  # Linear transformation
        return hidden_state

