from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    ComplementaryDataProcessorStep,
    EnvTransition,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    TransitionKey,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from torch import Tensor
from transformers import LlamaTokenizerFast

from .configuration_openvla_oft import OpenVLAOFTConfig

# The trailing space tokenizes to SentencePiece's "▁" (id 29871), which preceded the
# action tokens during training.
PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut: "


@ProcessorStepRegistry.register(name="openvla_oft_prompt")
@dataclass
class OpenVLAPromptProcessorStep(ComplementaryDataProcessorStep):
    """Wraps the lowercased task description in OpenVLA's prompt template."""

    task_key: str = "task"

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        task = complementary_data.get(self.task_key)
        if task is None:
            return complementary_data
        tasks = [task] if isinstance(task, str) else task
        prompts = [PROMPT_TEMPLATE.format(task=t.lower()) for t in tasks]
        return {**complementary_data, self.task_key: prompts[0] if isinstance(task, str) else prompts}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="openvla_oft_tokenizer")
@dataclass
class OpenVLATokenizerProcessorStep(TokenizerProcessorStep):
    """`TokenizerProcessorStep` that loads the Llama-2 tokenizer class directly.

    `AutoTokenizer` reads the repository's `config.json`, whose `auto_map` makes it
    ask whether to run OpenVLA's custom code.
    """

    def __post_init__(self) -> None:
        self.input_tokenizer = LlamaTokenizerFast.from_pretrained(self.tokenizer_name)


@ProcessorStepRegistry.register(name="openvla_oft_image_resize")
@dataclass
class OpenVLAImageResizeProcessorStep(ObservationProcessorStep):
    """Resizes every camera image to `size` x `size` with antialiased bicubic interpolation."""

    size: int = 224

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {key: self._resize(value) if _is_image(key) else value for key, value in observation.items()}

    def _resize(self, images: Tensor) -> Tensor:
        if images.shape[-2:] == (self.size, self.size):
            return images
        resized = F.interpolate(images, size=(self.size, self.size), mode="bicubic", antialias=True)
        return resized.clamp(0.0, 1.0)

    def get_config(self) -> dict[str, Any]:
        return {"size": self.size}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation = features[PipelineFeatureType.OBSERVATION]
        for key, feature in observation.items():
            if _is_image(key):
                observation[key] = PolicyFeature(
                    type=feature.type, shape=(feature.shape[0], self.size, self.size)
                )
        return features


def _is_image(key: str) -> bool:
    return key.startswith(f"{OBS_IMAGES}.")


@ProcessorStepRegistry.register(name="openvla_oft_clip")
@dataclass
class OpenVLAClipProcessorStep(ProcessorStep):
    """Clips the normalized state and action to [-1, 1].

    Placed after LeRobot's `QUANTILES` normalizer, this completes the original
    `BOUNDS_Q99` scheme, which clips after mapping [q01, q99] to [-1, 1].
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            transition[TransitionKey.OBSERVATION] = {
                **observation,
                OBS_STATE: observation[OBS_STATE].clamp(-1, 1),
            }
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition[TransitionKey.ACTION] = action.clamp(-1, 1)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="openvla_oft_libero_gripper")
@dataclass
class OpenVLALiberoGripperProcessorStep(ActionProcessorStep):
    """Converts the gripper action of the released LIBERO checkpoints to the LIBERO convention.

    The checkpoints predict the gripper on the original data loader's scale, 0 (close)
    to 1 (open). LIBERO expects -1 (open) or +1 (close), so the value is mapped to
    [-1, 1], binarized by its sign, and negated.
    """

    def action(self, action: PolicyAction) -> PolicyAction:
        gripper = -torch.sign(2 * action[..., -1:] - 1)
        return torch.cat([action[..., :-1], gripper], dim=-1)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_openvla_oft_pre_post_processors(
    config: OpenVLAOFTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the OpenVLA-OFT preprocessor and postprocessor.

    The preprocessor resizes the camera images, formats and tokenizes the prompt,
    moves data to the policy device, and normalizes and clips state and action. The
    postprocessor unnormalizes actions and moves them back to the CPU. The
    normalizer and unnormalizer are LeRobot's own steps, so `lerobot-train` can
    inject dataset statistics when fine-tuning from a pretrained policy.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        OpenVLAImageResizeProcessorStep(size=config.image_size),
        OpenVLAPromptProcessorStep(),
        OpenVLATokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name, padding="longest", padding_side="right"
        ),
        steps.to_device,
        steps.normalize,
        OpenVLAClipProcessorStep(),
    ]
    output_steps = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
