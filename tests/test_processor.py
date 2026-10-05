import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from lerobot_policy_openvla_oft import OpenVLAOFTConfig, make_openvla_oft_pre_post_processors
from lerobot_policy_openvla_oft.processor_openvla_oft import OpenVLAPromptProcessorStep

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
        device="cpu",
    )


def make_stats():
    stats = {"q01": torch.tensor([-1.0, 0.0]), "q99": torch.tensor([1.0, 2.0])}
    return {OBS_STATE: stats, ACTION: stats}


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
