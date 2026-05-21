"""Lightweight Ollama API client."""

from __future__ import annotations

import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config


ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class OllamaTool:
    """Small wrapper around the local Ollama HTTP API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        context_window: int = 2048,
        num_gpu: int | None = 0,
    ):
        if not base_url or not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be greater than 0")
        if context_window < 512:
            raise ValueError("context_window must be at least 512")
        if num_gpu is not None and num_gpu < 0:
            raise ValueError("num_gpu must be non-negative")

        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.context_window = context_window
        self.num_gpu = num_gpu

    @classmethod
    def from_config(cls, config_path: str = "model.yaml") -> "OllamaTool":
        """Build an Ollama tool from ``configs/model.yaml``."""
        load_dotenv(PROJECT_ROOT / ".env.example", override=False)
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        config = load_config(config_path)
        model_config = config.get("model", {})

        if not isinstance(model_config, dict):
            raise ValueError("model config must be a mapping")

        provider = str(model_config.get("provider", "ollama")).lower()
        if provider != "ollama":
            raise ValueError(f"Unsupported model provider for OllamaTool: {provider}")

        base_url = _resolve_config_value(
            model_config.get("base_url"), field_name="model.base_url"
        )
        model = _resolve_config_value(
            model_config.get("name"), field_name="model.name"
        )
        timeout_seconds = _resolve_int_config_value(
            model_config.get("timeout_seconds", 120),
            field_name="model.timeout_seconds",
        )
        context_window = _resolve_int_config_value(
            model_config.get("context_window", 2048),
            field_name="model.context_window",
        )
        num_gpu = _resolve_optional_int_config_value(
            model_config.get("num_gpu", 0),
            field_name="model.num_gpu",
        )
        return cls(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            context_window=context_window,
            num_gpu=num_gpu,
        )

    def check_connection(self) -> bool:
        """Return True when the local Ollama API is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Generate text from the configured Ollama model."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.context_window,
            },
        }
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Make sure Ollama is running locally."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError("Ollama request timed out.") from exc
        except requests.HTTPError as exc:
            response = exc.response
            body = ""
            if response is not None and response.text:
                body = f" Response body: {response.text[:500]}"
            hint = ""
            if "cudaMalloc" in body or "out of memory" in body:
                hint = " Try setting model.num_gpu to 0 or using a smaller model."
            elif "GGML_ASSERT" in body or "mem_buffer" in body:
                hint = " Try lowering model.context_window or using a smaller model."
            raise RuntimeError(
                f"Ollama API returned an error: {exc}{body}{hint}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama returned an invalid JSON response.") from exc

        generated_text = data.get("response")
        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama response did not contain generated text.")

        return generated_text.strip()


def _resolve_config_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required config value: {field_name}")

    raw_value = value.strip()
    match = ENV_REF_PATTERN.match(raw_value)
    if not match:
        return raw_value

    env_name = match.group(1)
    env_value = os.getenv(env_name)
    if not env_value:
        raise ValueError(
            f"Config value {field_name} references {env_name}, but it is not set"
        )

    return env_value.strip()


def _resolve_int_config_value(value: Any, field_name: str) -> int:
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        resolved = _resolve_config_value(value, field_name)
        try:
            return int(resolved)
        except ValueError as exc:
            raise ValueError(f"Config value must be an integer: {field_name}") from exc

    raise ValueError(f"Config value must be an integer: {field_name}")


def _resolve_optional_int_config_value(value: Any, field_name: str) -> int | None:
    if value is None:
        return None

    return _resolve_int_config_value(value, field_name)
