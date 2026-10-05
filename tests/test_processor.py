import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from lerobot_policy_openvla_oft import OpenVLAOFTConfig, make_openvla_oft_pre_post_processors
from lerobot_policy_openvla_oft.processor_openvla_oft import (
    OpenVLAImageResizeProcessorStep,
    OpenVLAPromptProcessorStep,
)

TASK = "Pick up the black bowl between the plate and the ramekin and place it on the plate"
# Recorded with the original stack (transformers 4.40.1, LlamaTokenizerFast) for
# f"In: What action should the robot take to {TASK.lower()}?\nOut: ".
EXPECTED_IDS = [
    1, 512, 29901, 1724, 3158, 881, 278, 19964, 2125, 304, 5839, 701, 278, 4628, 12580, 29880, 1546,
    278, 15284, 322, 278, 364, 420, 9089, 322, 2058, 372, 373, 278, 15284, 29973, 13, 3744, 29901, 29871,
]  # fmt: skip


def make_config():
    return OpenVLAOFTConfig(
        input_features={
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        action_norm_mask=[True, False],
        device="cpu",
    )


Q01, Q99 = torch.tensor([-0.5, 0.0]), torch.tensor([1.5, 1.0])
MASK = torch.tensor([True, False])


def make_stats():
    stats = {"q01": Q01, "q99": Q99}
    return {OBS_STATE: stats, ACTION: stats}


def original_normalize(x, mask):
    """`prismatic/vla/datasets/rlds/utils/data_utils.py:72-83` (BOUNDS_Q99)."""
    return torch.where(mask, torch.clamp(2 * (x - Q01) / (Q99 - Q01 + 1e-8) - 1, -1, 1), x)


def original_unnormalize(a, mask):
    """`prismatic/extern/hf/modeling_prismatic.py:785-789`."""
    return torch.where(mask, 0.5 * (a + 1) * (Q99 - Q01 + 1e-8) + Q01, a)


@pytest.fixture(scope="module")
def processors():
    return make_openvla_oft_pre_post_processors(make_config(), make_stats())


def test_prompt_step_lowercases_task_into_template():
    step = OpenVLAPromptProcessorStep()

    single = step.complementary_data({"task": "Open the Drawer"})
    batch = step.complementary_data({"task": ["Open the Drawer", "Close it"]})

    assert single["task"] == "In: What action should the robot take to open the drawer?\nOut: "
    assert batch["task"][1] == "In: What action should the robot take to close it?\nOut: "


def test_tokens_match_original_tokenizer(processors):
    preprocessor, _ = processors

    batch = preprocessor(
        {f"{OBS_IMAGES}.image": torch.rand(3, 224, 224), OBS_STATE: torch.zeros(2), "task": TASK}
    )

    assert batch[OBS_LANGUAGE_TOKENS].tolist() == [EXPECTED_IDS]
    assert batch[OBS_LANGUAGE_ATTENTION_MASK].all()


def test_batched_prompts_are_right_padded(processors):
    preprocessor, _ = processors

    batch = preprocessor(
        {
            f"{OBS_IMAGES}.image": torch.rand(2, 3, 224, 224),
            OBS_STATE: torch.zeros(2, 2),
            "task": [TASK, "Open the drawer"],
        }
    )

    mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
    assert batch[OBS_LANGUAGE_TOKENS][0].tolist() == EXPECTED_IDS
    assert mask[0].all()
    assert mask[1].sum() < mask.shape[1]
    assert mask[1].int().diff().le(0).all()
    assert (batch[OBS_LANGUAGE_TOKENS][:, 0] == 1).all()


def test_pipeline_round_trip(tmp_path, processors):
    preprocessor, _ = processors
    preprocessor.save_pretrained(tmp_path)

    loaded = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename=f"{preprocessor.name}.json")
    batch = loaded({f"{OBS_IMAGES}.image": torch.rand(3, 224, 224), OBS_STATE: torch.zeros(2), "task": TASK})

    assert batch[OBS_LANGUAGE_TOKENS].tolist() == [EXPECTED_IDS]


def make_observation(state, **extra):
    return {f"{OBS_IMAGES}.image": torch.rand(3, 224, 224), OBS_STATE: state, "task": TASK, **extra}


def test_normalization_matches_original_bounds_q99(processors):
    preprocessor, _ = processors
    raw = torch.tensor([[-2.0, 0.3], [0.1, 0.7], [3.0, 5.0]])

    batch = preprocessor(make_observation(raw[0], action=raw))

    torch.testing.assert_close(batch[ACTION], original_normalize(raw, MASK))
    torch.testing.assert_close(batch[OBS_STATE][0], original_normalize(raw[0], torch.tensor([True, True])))


def test_unnormalization_matches_original(processors):
    _, postprocessor = processors
    normalized = torch.tensor([[-1.0, 0.2], [0.5, 1.0], [1.3, -0.4]])

    torch.testing.assert_close(postprocessor(normalized), original_unnormalize(normalized, MASK))


def test_action_norm_mask_survives_round_trip(tmp_path, processors):
    _, postprocessor = processors
    postprocessor.save_pretrained(tmp_path)

    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{postprocessor.name}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )

    normalized = torch.tensor([[0.5, 0.25]])
    torch.testing.assert_close(loaded(normalized), original_unnormalize(normalized, MASK))


def test_resizes_camera_images_only(processors):
    preprocessor, _ = processors

    batch = preprocessor(make_observation(torch.zeros(2)) | {f"{OBS_IMAGES}.image": torch.rand(3, 256, 320)})

    assert batch[f"{OBS_IMAGES}.image"].shape == (1, 3, 224, 224)
    assert batch[f"{OBS_IMAGES}.image"].min() >= 0 and batch[f"{OBS_IMAGES}.image"].max() <= 1
    assert batch[OBS_STATE].shape == (1, 2)


def test_resize_keeps_images_already_at_size():
    images = torch.rand(1, 3, 224, 224)

    resized = OpenVLAImageResizeProcessorStep(size=224).observation({f"{OBS_IMAGES}.image": images})

    assert resized[f"{OBS_IMAGES}.image"] is images
