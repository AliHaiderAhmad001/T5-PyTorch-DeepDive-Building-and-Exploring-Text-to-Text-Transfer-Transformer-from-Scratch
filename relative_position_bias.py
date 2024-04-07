import torch
import torch.nn as nn

class RelativePositionBias(nn.Module):
    """
    Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are
    invalid.

    We use smaller buckets for small absolute relative_position and larger buckets
    for larger absolute relative_positions. All relative positions >=max_distance
    map to the same bucket. All relative positions <=-max_distance map to the
    same bucket. This should allow for more graceful generalization to longer
    sequences than the model has been trained on.

    Args:
        bidirectional (bool): Whether the attention is bidirectional.
        num_buckets (int): Number of buckets.
        max_distance (int): Maximum distance for relative positions.
        num_heads (int): Number of attention heads.

    # REFRANCE: https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    def __init__(self, config):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = config.bidirectional
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.num_heads = config.num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Translate relative position to a bucket number.

        Args:
            relative_position (torch.Tensor): Relative position tensor.
            bidirectional (bool): Whether the attention is bidirectional.
            num_buckets (int): Number of buckets.
            max_distance (int): Maximum distance for relative positions.

        Returns:
            torch.Tensor: Bucket number tensor.
        """
        ret = 0 * relative_position  # Initialized to zero to handle both positive and negative positions
        if bidirectional:
            num_buckets //= 2  # Halve the buckets for bidirectional case
            ret += (relative_position < 0).long() * num_buckets
            relative_position = relative_position.abs()
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Compute val_if_large with safe clamping within [0, num_buckets - 1]
        val_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            torch.log(torch.tensor(max_distance / max_exact, dtype=torch.float)) *
            (num_buckets - max_exact)
        ).long()
        val_if_large = torch.minimum(val_if_large, torch.tensor(num_buckets - 1, dtype=torch.long))

        # Combine small and large relative positions
        ret += torch.where(is_small, relative_position, val_if_large)

        return ret

    def compute_bias(self, qlen, klen):
        """
        Compute binned relative position bias.

        Args:
            qlen (int): Length of the query sequence.
            klen (int): Length of the key sequence.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )

        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)

        return values


    def forward(self, qlen, klen):
        """
        Forward pass.

        Args:
            qlen (int): Length of the query sequence.
            klen (int): Length of the key sequence.

        Returns:
            torch.Tensor: Relative position bias tensor.
        """

        return self.compute_bias(qlen, klen)
