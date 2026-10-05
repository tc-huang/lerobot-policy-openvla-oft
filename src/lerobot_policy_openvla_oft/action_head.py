from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block: x + ReLU(Linear(LayerNorm(x)))."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.linear(self.norm(x)))


class L1RegressionActionHead(nn.Module):
    """MLP that maps action-token hidden states to continuous, normalized actions.

    The hidden states of the `action_dim` tokens of one chunk step are concatenated
    and decoded into that step's action vector.
    """

    def __init__(self, hidden_size: int, action_dim: int, num_blocks: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.input_norm = nn.LayerNorm(action_dim * hidden_size)
        self.input_proj = nn.Linear(action_dim * hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList(ResidualBlock(hidden_size) for _ in range(num_blocks))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, action_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Maps (B, chunk_size * action_dim, hidden_size) to (B, chunk_size, action_dim)."""
        batch_size, _, hidden_size = hidden_states.shape
        x = hidden_states.reshape(batch_size, -1, self.action_dim * hidden_size)
        x = self.act(self.input_proj(self.input_norm(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.output_norm(x))
