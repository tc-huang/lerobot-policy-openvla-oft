import pytest
import torch
from tiny_models import TINY_IMAGE_SIZE, TINY_LLAMA, TINY_VIT

from lerobot_policy_openvla_oft.model import OpenVLAOFT
from lerobot_policy_openvla_oft.vision_backbone import FusedVisionBackbone

CHUNK_SIZE, ACTION_DIM, PROPRIO_DIM = 2, 3, 4
NUM_ACTION_TOKENS = CHUNK_SIZE * ACTION_DIM


@pytest.fixture
def model(tiny_vision, tiny_llm):
    torch.manual_seed(0)
    return OpenVLAOFT(tiny_vision, tiny_llm, CHUNK_SIZE, ACTION_DIM, PROPRIO_DIM).eval()


def film_model(tiny_llm, film_mask_padding):
    torch.manual_seed(0)
    vision = FusedVisionBackbone(TINY_IMAGE_SIZE, film_dim=TINY_LLAMA["hidden_size"], **TINY_VIT)
    return OpenVLAOFT(vision, tiny_llm, CHUNK_SIZE, ACTION_DIM, PROPRIO_DIM, film_mask_padding).eval()


def film_conditions(model, inputs):
    conditions = []
    model.vision.register_forward_hook(lambda module, args, output: conditions.append(args[1]))
    model(*inputs)
    return conditions[0]


def make_inputs(prompt_lengths, num_images=2):
    batch_size, max_length = len(prompt_lengths), max(prompt_lengths)
    prompt_mask = torch.arange(max_length) < torch.tensor(prompt_lengths)[:, None]
    input_ids = torch.randint(3, 64, (batch_size, max_length)).masked_fill(~prompt_mask, 0)
    input_ids[:, 0] = 1
    images = torch.rand(batch_size, num_images, 3, TINY_IMAGE_SIZE, TINY_IMAGE_SIZE)
    state = torch.randn(batch_size, PROPRIO_DIM)
    return images, input_ids, prompt_mask, state


def test_returns_one_hidden_state_per_action_token(model):
    hidden = model.action_hidden_states(*make_inputs([5, 5]))

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

    torch.testing.assert_close(model.action_hidden_states(images, input_ids, prompt_mask, state), expected)


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

    assert model(images, input_ids, prompt_mask).shape == (1, CHUNK_SIZE, ACTION_DIM)


def test_predicts_one_action_per_chunk_step(model):
    actions = model(*make_inputs([4, 6]))

    assert actions.shape == (2, CHUNK_SIZE, ACTION_DIM)


def test_every_parameter_receives_gradient(model):
    model.train()
    model(*make_inputs([4, 6])).sum().backward()

    assert [name for name, p in model.named_parameters() if p.grad is None] == []


def test_film_condition_averages_prompt_padding_and_eos_like_the_original(tiny_llm):
    model = film_model(tiny_llm, film_mask_padding=False)
    inputs = make_inputs([4, 7])
    input_ids = inputs[1]
    eos = torch.full((2, 1), model.llm.eos_token_id)

    expected = model.llm.embed(torch.cat([input_ids, eos], dim=1)).mean(dim=1)

    torch.testing.assert_close(film_conditions(model, inputs), expected)


def test_film_condition_can_exclude_padding(tiny_llm):
    model = film_model(tiny_llm, film_mask_padding=True)
    inputs = make_inputs([4, 7])
    input_ids, prompt_mask = inputs[1], inputs[2]
    eos = torch.tensor([model.llm.eos_token_id])

    expected = torch.stack(
        [
            model.llm.embed(torch.cat([ids[mask], eos])).mean(dim=0)
            for ids, mask in zip(input_ids, prompt_mask, strict=True)
        ]
    )

    torch.testing.assert_close(film_conditions(model, inputs), expected)


def test_every_film_parameter_receives_gradient(tiny_llm):
    model = film_model(tiny_llm, film_mask_padding=False).train()
    model(*make_inputs([4, 6])).sum().backward()

    assert [name for name, p in model.named_parameters() if p.grad is None] == []
    assert any(".film." in name for name, _ in model.named_parameters())
