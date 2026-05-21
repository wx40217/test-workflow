"""Provider-aware LangChain chat model factory."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional


class ProviderConfigError(ValueError):
    """Raised when model provider configuration is invalid."""


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    class_path: str
    default_base_url: Optional[str] = None
    default_model: Optional[str] = None
    default_use_responses_api: bool = False
    requires_base_url: bool = False
    supports_responses_api: bool = False
    supports_reasoning_effort: bool = False
    supports_tools_default: bool = True


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        class_path="langchain_openai.ChatOpenAI",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        default_use_responses_api=True,
        supports_responses_api=True,
        supports_reasoning_effort=True,
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        class_path="langchain_deepseek.ChatDeepSeek",
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
        default_use_responses_api=False,
        supports_reasoning_effort=True,
    ),
    "openai-compatible": ProviderSpec(
        name="openai-compatible",
        class_path="langchain_openai.ChatOpenAI",
        default_use_responses_api=False,
        requires_base_url=True,
        supports_reasoning_effort=True,
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        class_path="langchain_anthropic.ChatAnthropic",
        default_use_responses_api=False,
    ),
}


def supported_providers() -> list[str]:
    return sorted(PROVIDER_REGISTRY)


def normalize_provider(provider: Optional[str]) -> str:
    normalized = (provider or "openai").strip().lower()
    aliases = {
        "claude": "anthropic",
        "deepseek-v4": "deepseek",
        "openai_compatible": "openai-compatible",
        "compatible": "openai-compatible",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROVIDER_REGISTRY:
        raise ProviderConfigError(
            "未知模型 provider："
            f"{provider}。支持的 MODEL_PROVIDER：{', '.join(supported_providers())}"
        )
    return normalized


def provider_defaults(provider: Optional[str]) -> ProviderSpec:
    return PROVIDER_REGISTRY[normalize_provider(provider)]


def resolve_provider_defaults(config: Any) -> None:
    """Fill provider-derived defaults on a mutable ModelConfig-like object."""
    spec = provider_defaults(getattr(config, "provider", None))
    if not getattr(config, "provider", None):
        config.provider = spec.name
    if not getattr(config, "model_name", None) and spec.default_model:
        config.model_name = spec.default_model
    if not getattr(config, "base_url", None) and spec.default_base_url:
        config.base_url = spec.default_base_url
    if getattr(config, "use_responses_api", None) is None:
        config.use_responses_api = spec.default_use_responses_api
    if getattr(config, "supports_tools", None) is None:
        config.supports_tools = model_supports_tools(config)


def model_supports_tools(config: Any) -> bool:
    explicit = getattr(config, "supports_tools", None)
    if explicit is not None:
        return bool(explicit)

    provider = normalize_provider(getattr(config, "provider", None))
    model = (getattr(config, "model_name", "") or "").lower()
    if provider == "deepseek" and model == "deepseek-reasoner":
        return False
    if provider == "deepseek" and getattr(config, "thinking", None):
        return False
    return PROVIDER_REGISTRY[provider].supports_tools_default


def create_chat_model(config: Any) -> Any:
    """Create a provider-specific LangChain ChatModel with sanitized params."""
    resolve_provider_defaults(config)
    provider = normalize_provider(config.provider)
    spec = PROVIDER_REGISTRY[provider]
    _validate_config(config, spec)
    model_cls = _load_model_class(spec)

    if provider in {"openai", "openai-compatible"}:
        kwargs = _openai_kwargs(config, spec)
    elif provider == "deepseek":
        kwargs = _deepseek_kwargs(config)
    elif provider == "anthropic":
        kwargs = _anthropic_kwargs(config)
    else:
        raise ProviderConfigError(
            f"未实现的模型 provider：{provider}。支持列表：{', '.join(supported_providers())}"
        )

    return model_cls(**_drop_empty(kwargs))


def _validate_config(config: Any, spec: ProviderSpec) -> None:
    if not getattr(config, "api_key", None):
        raise ProviderConfigError(
            f"缺少模型 API key。请配置 MODEL_API_KEY，或为节点配置 {spec.name.upper()} 对应的 API key。"
        )
    if not getattr(config, "model_name", None):
        raise ProviderConfigError("缺少模型名称。请配置 MODEL_NAME 或节点级 *_MODEL_NAME。")
    if spec.requires_base_url and not getattr(config, "base_url", None):
        raise ProviderConfigError(
            "MODEL_PROVIDER=openai-compatible 需要配置 MODEL_BASE_URL 或节点级 *_BASE_URL。"
        )


def _load_model_class(spec: ProviderSpec) -> type[Any]:
    module_name, class_name = spec.class_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        package_name = module_name.replace("_", "-")
        raise ProviderConfigError(
            f"缺少 {spec.name} provider 依赖：请安装 {package_name}。"
        ) from exc
    return getattr(module, class_name)


def _openai_kwargs(config: Any, spec: ProviderSpec) -> dict[str, Any]:
    kwargs = {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "model": config.model_name,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "streaming": True,
        "temperature": config.temperature,
        "extra_body": dict(getattr(config, "extra_params", None) or {}),
    }
    if spec.supports_responses_api:
        kwargs["use_responses_api"] = config.use_responses_api
    if getattr(config, "reasoning_effort", None):
        kwargs.pop("temperature", None)
        kwargs["reasoning_effort"] = config.reasoning_effort
    return kwargs


def _deepseek_kwargs(config: Any) -> dict[str, Any]:
    extra_body = dict(getattr(config, "extra_params", None) or {})
    thinking = getattr(config, "thinking", None)
    if thinking is not None:
        extra_body["thinking"] = {"type": "enabled" if thinking else "disabled"}

    kwargs = {
        "api_key": config.api_key,
        "api_base": config.base_url,
        "model": config.model_name,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "streaming": True,
        "extra_body": extra_body,
    }
    if getattr(config, "reasoning_effort", None):
        kwargs["reasoning_effort"] = _deepseek_reasoning_effort(config.reasoning_effort)
    if not thinking and getattr(config, "temperature", None) is not None:
        kwargs["temperature"] = config.temperature
    return kwargs


def _deepseek_reasoning_effort(value: str) -> str:
    mapping = {
        "minimal": "high",
        "low": "high",
        "medium": "high",
        "high": "high",
        "max": "max",
        "xhigh": "max",
    }
    return mapping.get(value.strip().lower(), value)


def _anthropic_kwargs(config: Any) -> dict[str, Any]:
    kwargs = {
        "api_key": config.api_key,
        "model": config.model_name,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "streaming": True,
        "temperature": config.temperature,
    }
    kwargs.update(dict(getattr(config, "extra_params", None) or {}))
    return kwargs


def _drop_empty(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value not in (None, "", {})}
