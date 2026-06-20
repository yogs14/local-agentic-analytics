"""Lightweight Ollama API client."""

from __future__ import annotations

import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config


ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
DEFAULT_KEEP_ALIVE = "30m"
KEEP_ALIVE_PATTERN = re.compile(r"^[1-9][0-9]*[smh]$")
OLLAMA_DURATION_METRIC_KEYS = {
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
}
OLLAMA_COUNT_METRIC_KEYS = {
    "prompt_eval_count",
    "eval_count",
}


class OllamaTool:
    """Small wrapper around the local Ollama HTTP API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        context_window: int = 2048,
        num_gpu: int | None = 0,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
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
        keep_alive = _normalize_keep_alive(keep_alive)

        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.context_window = context_window
        self.num_gpu = num_gpu
        self.keep_alive = keep_alive
        self.last_metrics: dict[str, float | int] = {}

    @classmethod
    def from_config(cls, config_path: str = "model.yaml") -> "OllamaTool":
        """Build an Ollama tool from ``configs/model.yaml``."""
        load_dotenv(PROJECT_ROOT / ".env.example", override=False)
        load_dotenv(PROJECT_ROOT / ".env", override=False)
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
        keep_alive = _resolve_keep_alive_config_value(
            model_config.get("keep_alive", DEFAULT_KEEP_ALIVE),
            field_name="model.keep_alive",
        )
        return cls(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            context_window=context_window,
            num_gpu=num_gpu,
            keep_alive=keep_alive,
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

        self.last_metrics = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
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
            self.last_metrics = _extract_ollama_metrics(data)
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

    def get_last_metrics(self) -> dict[str, float | int]:
        """Return profiling metrics from the most recent Ollama generation."""
        return dict(self.last_metrics)

    def unload(self, model: str | None = None) -> bool:
        """Unload a model from memory by requesting ``keep_alive=0``.

        Used by benchmark scaffolding to free VRAM between runs. Best-effort:
        returns ``True`` on success, ``False`` if the request fails, and never
        raises so it cannot abort a benchmark.
        """
        target = (model or self.model).strip()
        if not target:
            return False

        payload = {"model": target, "keep_alive": 0}
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException:
            return False
        return True


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


def _resolve_keep_alive_config_value(value: Any, field_name: str) -> str:
    if value is None:
        return DEFAULT_KEEP_ALIVE

    if not isinstance(value, str):
        raise ValueError(f"Config value must be a duration string: {field_name}")

    resolved = value.strip()
    if not resolved:
        return DEFAULT_KEEP_ALIVE
    if ENV_REF_PATTERN.match(resolved):
        resolved = _resolve_config_value(resolved, field_name)

    return _normalize_keep_alive(resolved)


def _normalize_keep_alive(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_KEEP_ALIVE

    keep_alive = value.strip()
    if not KEEP_ALIVE_PATTERN.match(keep_alive):
        raise ValueError(
            "keep_alive must be a duration string such as '5m', '30m', or '1h'"
        )
    return keep_alive


def _extract_ollama_metrics(data: dict[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    for key in OLLAMA_DURATION_METRIC_KEYS:
        if key in data:
            seconds = _duration_nanoseconds_to_seconds(data.get(key))
            if seconds is not None:
                metrics[key] = seconds

    for key in OLLAMA_COUNT_METRIC_KEYS:
        if key in data:
            count = _coerce_int(data.get(key))
            if count is not None:
                metrics[key] = count

    return metrics


def _duration_nanoseconds_to_seconds(value: Any) -> float | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return number / 1_000_000_000


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None
