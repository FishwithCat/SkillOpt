from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest


class _OpenAIClientStub:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


if "openai" not in sys.modules and importlib.util.find_spec("openai") is None:
    openai_stub = types.ModuleType("openai")
    openai_stub.AzureOpenAI = _OpenAIClientStub
    openai_stub.OpenAI = _OpenAIClientStub
    sys.modules["openai"] = openai_stub

from skillopt.model import (
    azure_openai,
    chat_target,
    configure_deepseek_chat,
    get_target_backend,
    set_target_backend,
    set_target_deployment,
)
from skillopt.model.common import default_model_for_backend, normalize_backend_name


def _args(*options: str) -> Any:
    return SimpleNamespace(config="configs/searchqa/default.yaml", cfg_options=list(options))


class _ChatCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


class _Client:
    def __init__(self) -> None:
        self.chat_completions = _ChatCompletions()
        self.chat = SimpleNamespace(completions=self.chat_completions)


def test_deepseek_backend_uses_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    original_backend = get_target_backend()

    def fake_openai(**kwargs: Any) -> _Client:
        seen.update(kwargs)
        return _Client()

    monkeypatch.setattr(azure_openai, "OpenAI", fake_openai)
    try:
        configure_deepseek_chat(base_url="https://deepseek.test", api_key="secret")
        set_target_backend("deepseek_chat")
        set_target_deployment(default_model_for_backend("deepseek_chat"))

        text, usage = chat_target("system", "user", max_completion_tokens=123, retries=1, reasoning_effort="Max")
        client = azure_openai.get_target_client()
    finally:
        set_target_backend(original_backend)

    assert text == "ok"
    assert usage["total_tokens"] == 5
    assert seen["base_url"] == "https://deepseek.test"
    assert seen["api_key"] == "secret"
    assert client.chat_completions.kwargs["model"] == "deepseek-v4-pro"
    assert client.chat_completions.kwargs["max_tokens"] == 123
    assert client.chat_completions.kwargs["reasoning_effort"] == "max"
    assert client.chat_completions.kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "max_completion_tokens" not in client.chat_completions.kwargs
    assert normalize_backend_name("deepseek") == "deepseek_chat"


def test_deepseek_config_overrides_base_openai_defaults() -> None:
    from scripts.train import load_config

    cfg = load_config(_args("model.backend=deepseek"))

    assert cfg["optimizer_backend"] == "deepseek_chat"
    assert cfg["target_backend"] == "deepseek_chat"
    assert cfg["optimizer_model"] == "deepseek-v4-pro"
    assert cfg["target_model"] == "deepseek-v4-pro"
    assert cfg["reasoning_effort"] == "max"
    assert cfg["deepseek_base_url"] == "https://api.deepseek.com"
