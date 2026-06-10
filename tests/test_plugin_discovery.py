import importlib
import importlib.metadata as metadata

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import _get_policy_cls_from_policy_name
from lerobot.utils.import_utils import register_third_party_plugins


PLUGIN_DISTRIBUTION = "lerobot_policy_openvla_oft"
PLUGIN_POLICY_TYPE = "openvla_oft"


def test_register_third_party_plugins_discovers_openvla_oft_policy():
    installed_policy_plugins = {
        dist_name
        for dist in metadata.distributions()
        if (dist_name := dist.metadata.get("Name", "")).startswith("lerobot_policy_")
    }

    assert PLUGIN_DISTRIBUTION in installed_policy_plugins

    register_third_party_plugins()

    assert PLUGIN_POLICY_TYPE in PreTrainedConfig.get_known_choices()
    config_cls = PreTrainedConfig.get_choice_class(PLUGIN_POLICY_TYPE)
    assert config_cls.__name__ == "OpenvlaOftConfig"
    assert config_cls.__module__ == (f"{PLUGIN_DISTRIBUTION}.configuration_openvla_oft")

    plugin_module = importlib.import_module(PLUGIN_DISTRIBUTION)
    assert plugin_module.OpenvlaOftConfig is config_cls
    assert plugin_module.OpenvlaOftPolicy.__name__ == "OpenvlaOftPolicy"
    assert (
        plugin_module.make_openvla_oft_pre_post_processors.__name__
        == "make_openvla_oft_pre_post_processors"
    )

    policy_cls = _get_policy_cls_from_policy_name(PLUGIN_POLICY_TYPE)
    assert policy_cls is plugin_module.OpenvlaOftPolicy
    assert policy_cls.name == PLUGIN_POLICY_TYPE
    assert policy_cls.config_class is config_cls

    processor_module = importlib.import_module(
        config_cls.__module__.replace("configuration_", "processor_")
    )
    processor_factory = getattr(
        processor_module, f"make_{PLUGIN_POLICY_TYPE}_pre_post_processors"
    )
    assert processor_factory is plugin_module.make_openvla_oft_pre_post_processors
    assert callable(processor_factory)
