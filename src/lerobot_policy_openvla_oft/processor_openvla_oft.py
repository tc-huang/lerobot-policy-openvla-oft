from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.normalize_processor import _NormalizationMixin
from lerobot.utils.constants import OBS_IMAGES
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


@dataclass
class _BoundsQ99Mixin(_NormalizationMixin):
    """Turns LeRobot's `QUANTILES` mode into the original `BOUNDS_Q99` scheme.

    On top of mapping [q01, q99] to [-1, 1], normalized values are clipped to
    [-1, 1], and action dimensions whose `action_norm_mask` entry is False are
    passed through unchanged in both directions.
    """

    action_norm_mask: list[bool] | None = None

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:
        result = super()._apply_transform(tensor, key, feature_type, inverse=inverse)
        if self.norm_map.get(feature_type) != NormalizationMode.QUANTILES:
            return result
        if not inverse:
            result = result.clamp(-1.0, 1.0)
        if feature_type == FeatureType.ACTION and self.action_norm_mask is not None:
            mask = torch.tensor(self.action_norm_mask, device=tensor.device)
            result = torch.where(mask, result, tensor)
        return result

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "action_norm_mask": self.action_norm_mask}


@ProcessorStepRegistry.register(name="openvla_oft_normalizer")
@dataclass
class OpenVLANormalizerProcessorStep(_BoundsQ99Mixin, NormalizerProcessorStep):
    """Normalizes state and action with the original `BOUNDS_Q99` scheme."""


@ProcessorStepRegistry.register(name="openvla_oft_unnormalizer")
@dataclass
class OpenVLAUnnormalizerProcessorStep(_BoundsQ99Mixin, UnnormalizerProcessorStep):
    """Unnormalizes actions with the original `BOUNDS_Q99` scheme."""


def make_openvla_oft_pre_post_processors(
    config: OpenVLAOFTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the OpenVLA-OFT preprocessor and postprocessor.

    The preprocessor resizes the camera images, formats and tokenizes the prompt,
    moves data to the policy device, and normalizes state and action. The
    postprocessor unnormalizes actions and moves them back to the CPU.
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
        OpenVLANormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            action_norm_mask=config.action_norm_mask,
        ),
    ]
    output_steps = [
        OpenVLAUnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            action_norm_mask=config.action_norm_mask,
        ),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
