import torch

from lerobot_policy_openvla_oft.action_head import L1RegressionActionHead

CHUNK_SIZE, ACTION_DIM, HIDDEN_SIZE = 3, 2, 8


def test_maps_hidden_states_to_action_chunk():
    head = L1RegressionActionHead(HIDDEN_SIZE, ACTION_DIM)

    actions = head(torch.randn(4, CHUNK_SIZE * ACTION_DIM, HIDDEN_SIZE))

    assert actions.shape == (4, CHUNK_SIZE, ACTION_DIM)


def test_each_step_is_decoded_from_its_own_tokens():
    head = L1RegressionActionHead(HIDDEN_SIZE, ACTION_DIM)
    hidden = torch.randn(1, CHUNK_SIZE * ACTION_DIM, HIDDEN_SIZE)
    changed = hidden.clone()
    changed[:, ACTION_DIM : 2 * ACTION_DIM] += torch.randn(1, ACTION_DIM, HIDDEN_SIZE)

    actions, actions_after_change = head(hidden), head(changed)

    assert not torch.allclose(actions[:, 1], actions_after_change[:, 1])
    torch.testing.assert_close(actions[:, [0, 2]], actions_after_change[:, [0, 2]])


def test_size_matches_paper():
    with torch.device("meta"):
        head = L1RegressionActionHead(hidden_size=4096, action_dim=7)

    assert round(sum(p.numel() for p in head.parameters()) / 1e6) == 151
