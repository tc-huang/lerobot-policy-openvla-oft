"""Builds state dicts in the layout of the released OpenVLA-OFT checkpoints."""

import torch

RELEASED_ACTION_HEAD_LAYERS = {
    "input_norm": "layer_norm1",
    "input_proj": "fc1",
    "output_norm": "layer_norm2",
    "output_proj": "fc2",
}


def released_key(key):
    """Inverse of the conversion, written independently from the released module names."""
    if key.startswith("action_head.blocks."):
        _, _, index, layer, param = key.split(".")
        return f"module.model.mlp_resnet_blocks.{index}.ffn.{0 if layer == 'norm' else 1}.{param}"
    if key.startswith("action_head."):
        _, layer, param = key.split(".")
        return f"module.model.{RELEASED_ACTION_HEAD_LAYERS[layer]}.{param}"
    if key.startswith("proprio_projector."):
        return "module." + key.removeprefix("proprio_projector.")
    key = key.replace("vision.dinov2.vit.", "vision_backbone.featurizer.")
    key = key.replace("vision.siglip.vit.", "vision_backbone.fused_featurizer.")
    key = key.replace("vision_projector.", "projector.").replace("llm.model.", "language_model.model.")
    return key.replace(".ls1.gamma", ".ls1.scale_factor").replace(".ls2.gamma", ".ls2.scale_factor")


def released_checkpoint(model):
    vla, action_head, proprio = {}, {}, {}
    for key, value in model.state_dict().items():
        target = (
            action_head if key.startswith("action_head.") else proprio if key.startswith("proprio") else vla
        )
        target[released_key(key)] = value.clone()
    pruned_block = len(model.vision.dinov2.vit.blocks)
    vla |= {
        "language_model.lm_head.weight": torch.zeros(1),
        f"vision_backbone.featurizer.blocks.{pruned_block}.attn.qkv.weight": torch.zeros(1),
        "vision_backbone.featurizer.norm.weight": torch.zeros(1),
        "vision_backbone.fused_featurizer.attn_pool.latent": torch.zeros(1),
    }
    return vla, action_head, proprio
