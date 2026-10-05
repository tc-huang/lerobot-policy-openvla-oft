from dataclasses import dataclass
from typing import Any

import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
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


def make_openvla_oft_pre_post_processors(
    config: OpenVLAOFTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Builds the OpenVLA-OFT preprocessor and postprocessor.

    The preprocessor formats and tokenizes the prompt, moves data to the policy
    device, and normalizes state and action. The postprocessor unnormalizes actions
    and moves them back to the CPU.
    """
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        OpenVLAPromptProcessorStep(),
        OpenVLATokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name, padding="longest", padding_side="right"
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [steps.unnormalize, steps.to_cpu]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
