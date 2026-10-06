import torch
from torch import Tensor, nn

from .action_head import L1RegressionActionHead
from .language_model import BidirectionalLlama
from .projectors import ProprioProjector, VisionProjector
from .vision_backbone import FusedVisionBackbone


class OpenVLAOFT(nn.Module):
    """OpenVLA-OFT network that predicts a whole action chunk in one forward pass.

    The LLM input sequence is laid out as

        [BOS] [image patches] [proprio] [prompt] [action placeholders] [EOS] [padding]

    with one placeholder per action dimension per chunk step. Placeholders are zero
    vectors, so they differ only through their rotary position; bidirectional
    attention lets every placeholder read the whole sequence, and an MLP action head
    regresses the actions from the placeholders' hidden states.

    If the vision backbone uses FiLM, it is conditioned on the average embedding of
    the prompt tokens and EOS. Like the original, the average includes padding
    unless `film_mask_padding` is set.
    """

    def __init__(
        self,
        vision: FusedVisionBackbone,
        llm: BidirectionalLlama,
        chunk_size: int,
        action_dim: int,
        proprio_dim: int | None,
        film_mask_padding: bool = False,
    ):
        super().__init__()
        self.vision = vision
        self.vision_projector = VisionProjector(vision.embed_dim, llm.hidden_size)
        self.proprio_projector = ProprioProjector(proprio_dim, llm.hidden_size) if proprio_dim else None
        self.llm = llm
        self.action_head = L1RegressionActionHead(llm.hidden_size, action_dim)
        self.num_action_tokens = chunk_size * action_dim
        self.film_mask_padding = film_mask_padding

    def forward(
        self, images: Tensor, input_ids: Tensor, prompt_mask: Tensor, state: Tensor | None = None
    ) -> Tensor:
        """Predicts a normalized action chunk of shape (B, chunk_size, action_dim).

        Arguments are the same as for `action_hidden_states`.
        """
        return self.action_head(self.action_hidden_states(images, input_ids, prompt_mask, state))

    def action_hidden_states(
        self, images: Tensor, input_ids: Tensor, prompt_mask: Tensor, state: Tensor | None = None
    ) -> Tensor:
        """Returns the hidden states that decode the action chunk.

        Args:
            images: (B, num_images, 3, H, W) tensor with values in [0, 1].
            input_ids: (B, L) right-padded prompt token ids starting with BOS.
            prompt_mask: (B, L) boolean tensor, False for padding.
            state: (B, proprio_dim) normalized robot state, required if the model
                has a proprio projector.

        Returns:
            (B, chunk_size * action_dim, hidden_size) hidden states, one per action
            dimension per chunk step.
        """
        condition = self._film_condition(input_ids, prompt_mask) if self.vision.uses_film else None
        prefix = self._embed_prefix(input_ids[:, :1], images, state, condition)
        body, body_mask = self._embed_body(input_ids[:, 1:], prompt_mask[:, 1:])
        prefix_mask = prompt_mask.new_ones(prefix.shape[:2])
        hidden = self.llm(torch.cat([prefix, body], dim=1), torch.cat([prefix_mask, body_mask], dim=1))

        # As in autoregressive OpenVLA, each action token is read from the position
        # right before it, starting at the last prompt token.
        first_action = prefix.shape[1] + prompt_mask[:, 1:].sum(dim=1)
        index = (first_action - 1)[:, None] + torch.arange(self.num_action_tokens, device=hidden.device)
        return hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))

    def _film_condition(self, input_ids: Tensor, prompt_mask: Tensor) -> Tensor:
        eos = input_ids.new_full((input_ids.shape[0], 1), self.llm.eos_token_id)
        embeddings = self.llm.embed(torch.cat([input_ids, eos], dim=1))
        if not self.film_mask_padding:
            return embeddings.mean(dim=1)
        mask = torch.cat([prompt_mask, prompt_mask.new_ones(eos.shape)], dim=1).unsqueeze(-1)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1)

    def _embed_prefix(
        self, bos_ids: Tensor, images: Tensor, state: Tensor | None, condition: Tensor | None
    ) -> Tensor:
        tokens = [self.llm.embed(bos_ids), self.vision_projector(self.vision(images, condition))]
        if self.proprio_projector is not None:
            tokens.append(self.proprio_projector(state))
        return torch.cat(tokens, dim=1)

    def _embed_body(self, prompt_ids: Tensor, prompt_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Appends placeholders and EOS to each prompt, then moves padding to the end."""
        batch_size = prompt_ids.shape[0]
        prompt = self.llm.embed(prompt_ids)
        placeholders = prompt.new_zeros(batch_size, self.num_action_tokens, prompt.shape[-1])
        eos = self.llm.embed(prompt_ids.new_full((batch_size, 1), self.llm.eos_token_id))
        body = torch.cat([prompt, placeholders, eos], dim=1)
        body_mask = torch.cat(
            [prompt_mask, prompt_mask.new_ones(batch_size, self.num_action_tokens + 1)], dim=1
        )

        order = torch.argsort((~body_mask).to(torch.uint8), dim=1, stable=True)
        body = body.gather(1, order[..., None].expand_as(body))
        return body, body_mask.gather(1, order)
