from torch import Tensor, nn


class VisionProjector(nn.Module):
    """3-layer GELU MLP that maps fused patch features into the LLM embedding space."""

    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim)
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim)
        self.fc3 = nn.Linear(llm_dim, llm_dim)
        self.act = nn.GELU()

    def forward(self, patches: Tensor) -> Tensor:
        """Maps (B, num_patches, vision_dim) to (B, num_patches, llm_dim)."""
        return self.fc3(self.act(self.fc2(self.act(self.fc1(patches)))))


class ProprioProjector(nn.Module):
    """2-layer GELU MLP that maps the robot state to a single LLM token embedding."""

    def __init__(self, proprio_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act = nn.GELU()

    def forward(self, state: Tensor) -> Tensor:
        """Maps (B, proprio_dim) to (B, 1, llm_dim)."""
        return self.fc2(self.act(self.fc1(state))).unsqueeze(1)
