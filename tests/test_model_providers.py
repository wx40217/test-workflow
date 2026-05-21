import os
import unittest
from unittest.mock import patch

from config.settings import ModelConfig, Settings
from src.llm import providers
from src.llm.providers import ProviderConfigError, create_chat_model
from src.workflow.graph import create_workflow


class FakeChatModel:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


class ProviderFactoryTests(unittest.TestCase):
    def setUp(self):
        FakeChatModel.calls.clear()

    def test_openai_uses_chatopenai_with_responses_api_default(self):
        config = ModelConfig(api_key="test", model_name="gpt-4o", provider="openai")

        with patch.object(providers, "_load_model_class", return_value=FakeChatModel):
            model = create_chat_model(config)

        self.assertIsInstance(model, FakeChatModel)
        self.assertEqual(model.kwargs["base_url"], "https://api.openai.com/v1")
        self.assertTrue(model.kwargs["use_responses_api"])

    def test_deepseek_uses_chatdeepseek_defaults_and_not_responses_api(self):
        config = ModelConfig(api_key="test", provider="deepseek")

        with patch.object(providers, "_load_model_class", return_value=FakeChatModel):
            model = create_chat_model(config)

        self.assertEqual(config.model_name, "deepseek-v4-flash")
        self.assertEqual(model.kwargs["api_base"], "https://api.deepseek.com")
        self.assertNotIn("use_responses_api", model.kwargs)

    def test_deepseek_thinking_uses_extra_body_and_omits_temperature(self):
        config = ModelConfig(
            api_key="test",
            provider="deepseek",
            model_name="deepseek-v4-pro",
            thinking=True,
            reasoning_effort="medium",
        )

        with patch.object(providers, "_load_model_class", return_value=FakeChatModel):
            model = create_chat_model(config)

        self.assertEqual(model.kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(model.kwargs["reasoning_effort"], "high")
        self.assertNotIn("temperature", model.kwargs)

    def test_deepseek_thinking_marks_tools_unsupported_for_react_guard(self):
        config = ModelConfig(
            api_key="test",
            provider="deepseek",
            model_name="deepseek-v4-pro",
            thinking=True,
        )

        self.assertFalse(config.supports_tools)

    def test_openai_compatible_requires_base_url(self):
        config = ModelConfig(
            api_key="test",
            provider="openai-compatible",
            model_name="compatible-model",
        )

        with self.assertRaisesRegex(ProviderConfigError, "MODEL_BASE_URL"):
            create_chat_model(config)

    def test_anthropic_omits_openai_specific_params(self):
        config = ModelConfig(
            api_key="test",
            provider="anthropic",
            model_name="claude-3-5-sonnet-latest",
            base_url="https://should-not-pass.example",
            use_responses_api=True,
            reasoning_effort="high",
        )

        with patch.object(providers, "_load_model_class", return_value=FakeChatModel):
            model = create_chat_model(config)

        self.assertNotIn("base_url", model.kwargs)
        self.assertNotIn("use_responses_api", model.kwargs)
        self.assertNotIn("reasoning_effort", model.kwargs)

    def test_unknown_provider_mentions_supported_list(self):
        with self.assertRaisesRegex(ProviderConfigError, "openai-compatible"):
            ModelConfig(api_key="test", provider="unknown")


class ProviderSettingsTests(unittest.TestCase):
    def test_global_model_env_creates_four_node_configs(self):
        env = {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_API_KEY": "test-key",
            "MODEL_NAME": "deepseek-v4-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        configs = [
            settings.get_generator_config(),
            settings.get_reviewer_config(),
            settings.get_optimizer_config(),
            settings.get_analyzer_config(),
        ]
        self.assertTrue(all(config.provider == "deepseek" for config in configs))
        self.assertTrue(all(config.api_key == "test-key" for config in configs))
        self.assertTrue(all(config.base_url == "https://api.deepseek.com" for config in configs))

    def test_node_level_model_name_only_overrides_generator(self):
        env = {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_API_KEY": "test-key",
            "MODEL_NAME": "deepseek-v4-flash",
            "GENERATOR_MODEL_NAME": "deepseek-v4-pro",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.get_generator_config().model_name, "deepseek-v4-pro")
        self.assertEqual(settings.get_reviewer_config().model_name, "deepseek-v4-flash")

    def test_legacy_model_config_constructor_still_works(self):
        config = ModelConfig(
            api_key="test",
            base_url="http://example.invalid",
            model_name="fake",
        )

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.base_url, "http://example.invalid")
        self.assertEqual(config.model_name, "fake")

    def test_create_workflow_keeps_legacy_explicit_options(self):
        workflow = create_workflow(
            api_key="test",
            base_url="http://example.invalid",
            generator_model="fake-generator",
            reviewer_model="fake-reviewer",
            optimizer_model="fake-optimizer",
        )

        self.assertEqual(workflow.generator.config.api_key, "test")
        self.assertEqual(workflow.generator.config.base_url, "http://example.invalid")
        self.assertEqual(workflow.generator.config.model_name, "fake-generator")
        self.assertEqual(workflow.reviewer.config.model_name, "fake-reviewer")
        self.assertEqual(workflow.optimizer.config.model_name, "fake-optimizer")


if __name__ == "__main__":
    unittest.main()
