import pytest
import torch
from tiny_models import TINY_IMAGE_SIZE

from lerobot_policy_openvla_oft.model import OpenVLAOFT

CHUNK_SIZE, ACTION_DIM, PROPRIO_DIM = 2, 3, 4
NUM_ACTION_TOKENS = CHUNK_SIZE * ACTION_DIM


@pytest.fixture
def model(tiny_vision, tiny_llm):
    torch.manual_seed(0)
    return OpenVLAOFT(tiny_vision, tiny_llm, CHUNK_SIZE, ACTION_DIM, PROPRIO_DIM).eval()


def make_inputs(prompt_lengths, num_images=2):
    batch_size, max_length = len(prompt_lengths), max(prompt_lengths)
    prompt_mask = torch.arange(max_length) < torch.tensor(prompt_lengths)[:, None]
    input_ids = torch.randint(3, 64, (batch_size, max_length)).masked_fill(~prompt_mask, 0)
    input_ids[:, 0] = 1
    images = torch.rand(batch_size, num_images, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)
    state = torch.randn(batch_size, PROPRIO_DIM)
    return images, input_ids, prompt_mask, state


def test_returns_one_hidden_state_per_action_token(model):
    hidden = model(*make_inputs([5, 5]))

    assert hidden.shape == (2, NUM_ACTION_TOKENS, model.llm.hidden_size)


def test_reads_hidden_states_one_position_before_each_placeholder(model):
    images, input_ids, prompt_mask, state = make_inputs([5])
    embed = model.llm.embed
    sequence = torch.cat(
        [
            embed(input_ids[:, :1]),
            model.vision_projector(model.vision(images)),
            model.proprio_projector(state),
            embed(input_ids[:, 1:]),
            torch.zeros(1, NUM_ACTION_TOKENS, model.llm.hidden_size),
            embed(torch.tensor([[model.llm.eos_token_id]])),
        ],
        dim=1,
    )
    hidden = model.llm(sequence, torch.ones(sequence.shape[:2], dtype=torch.bool))
    last_prompt = sequence.shape[1] - NUM_ACTION_TOKENS - 2

    expected = hidden[:, last_prompt : last_prompt + NUM_ACTION_TOKENS]

    torch.testing.assert_close(model(images, input_ids, prompt_mask, state), expected)


def test_padded_batch_matches_unbatched(model):
    images, input_ids, prompt_mask, state = make_inputs([4, 7])

    batched = model(images, input_ids, prompt_mask, state)

    for i in range(2):
        length = int(prompt_mask[i].sum())
        single = model(
            images[i : i + 1],
            input_ids[i : i + 1, :length],
            prompt_mask[i : i + 1, :length],
            state[i : i + 1],
        )
        torch.testing.assert_close(batched[i : i + 1], single)


def test_runs_without_proprio(tiny_vision, tiny_llm):
    model = OpenVLAOFT(tiny_vision, tiny_llm, CHUNK_SIZE, ACTION_DIM, proprio_dim=None)
    images, input_ids, prompt_mask, _ = make_inputs([5])

    assert model(images, input_ids, prompt_mask).shape == (1, NUM_ACTION_TOKENS, model.llm.hidden_size)
