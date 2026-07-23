"""Teacher clients for synthetic insight generation.

``TeacherClient`` mirrors the project's ``OllamaTool.generate`` signature so it
can be injected exactly like the existing fake-client tests. Two implementations
ship here:

- ``FakeTeacherClient`` -- offline, deterministic, synthesizes a grounded
  narrative by reading the ``<STATS>`` block out of the prompt.
- ``OllamaTeacherClient`` -- talks to a real Ollama endpoint (configured via env
  vars) with retry and rate-limit handling.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Protocol, runtime_checkable

from local_agentic_analytics.finetune.narrative_synth import synthesize_from_block
from local_agentic_analytics.finetune.teacher_prompt import extract_stats_block


@runtime_checkable
class TeacherClient(Protocol):
    """Minimal text-generation contract shared by fake and real teachers."""

    def generate(
        self, prompt: str, temperature: float = 0.4, max_tokens: int = 384
    ) -> str: ...


class FakeTeacherClient:
    """Offline teacher that returns grounded narratives for tests/smoke runs."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, prompt: str, temperature: float = 0.4, max_tokens: int = 384
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
        )
        block = extract_stats_block(prompt)
        if self.seed is None:
            rng = random.Random(hash(block) & 0xFFFFFFFF)
        else:
            # Deterministic but block-dependent so distinct variants vary.
            rng = random.Random(self.seed ^ (hash(block) & 0xFFFFFFFF))
        return synthesize_from_block(block, rng)


# --- Real Ollama teacher -------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma2:2b"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OllamaTeacherClient:
    """Teacher backed by a real Ollama endpoint with retry + backoff."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        timeout_seconds: int = 180,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        max_backoff: float = 30.0,
        sleep: Any = time.sleep,
    ):
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self._sleep = sleep

    @classmethod
    def from_env(cls, **overrides: Any) -> "OllamaTeacherClient":
        """Build a client from ``TEACHER_*`` environment variables."""
        base_url = os.getenv("TEACHER_BASE_URL") or os.getenv(
            "TEACHER_ENDPOINT", DEFAULT_BASE_URL
        )
        model = os.getenv("TEACHER_MODEL", DEFAULT_MODEL)
        kwargs: dict[str, Any] = {"base_url": base_url, "model": model}
        timeout = os.getenv("TEACHER_TIMEOUT")
        if timeout:
            kwargs["timeout_seconds"] = int(timeout)
        retries = os.getenv("TEACHER_MAX_RETRIES")
        if retries:
            kwargs["max_retries"] = int(retries)
        kwargs.update(overrides)
        return cls(**kwargs)

    def _backoff_seconds(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_backoff)
        return min(self.backoff_base * (2**attempt), self.max_backoff)

    def generate(
        self, prompt: str, temperature: float = 0.4, max_tokens: int = 384
    ) -> str:
        # Imported lazily so offline tests never require the dependency.
        import requests

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        url = f"{self.base_url}/api/generate"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                if response.status_code in _RETRYABLE_STATUS:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    last_error = RuntimeError(
                        f"Teacher endpoint returned {response.status_code}"
                    )
                    if attempt < self.max_retries:
                        self._sleep(self._backoff_seconds(attempt, retry_after))
                        continue
                    raise last_error
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(self._backoff_seconds(attempt, None))
                    continue
                raise RuntimeError(f"Teacher request failed: {exc}") from exc
            else:
                text = data.get("response")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Teacher response did not contain text.")
                return text.strip()

        raise RuntimeError(f"Teacher request exhausted retries: {last_error}")


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --- Claude API teacher (Opus 4.8) ---------------------------------------------

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"


class AnthropicTeacherClient:
    """Teacher backed by the Claude API (default model: Opus 4.8).

    Notes specific to Opus 4.8 (see the claude-api reference):
    - ``temperature`` / ``top_p`` / ``top_k`` are removed and return a 400, so the
      ``temperature`` argument from the shared interface is accepted but NOT sent.
    - ``thinking`` is omitted (this is a short, well-specified transform, and the
      prompt already instructs final-answer-only output).
    - The official ``anthropic`` SDK auto-retries 429/5xx with backoff; we only
      raise the retry ceiling and guard the ``refusal`` stop reason.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        *,
        max_tokens: int = 512,
        max_retries: int = 4,
    ):
        if not model.strip():
            raise ValueError("model must not be empty")
        self.model = model.strip()
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = client

    @classmethod
    def from_env(cls, **overrides: Any) -> "AnthropicTeacherClient":
        """Build a client from environment configuration.

        Reads the model from ``TEACHER_MODEL`` (default ``claude-opus-4-8``) and the
        retry ceiling from ``TEACHER_MAX_RETRIES``. The API key is resolved by the
        SDK from ``ANTHROPIC_API_KEY`` (or an ``ant`` profile); the base URL from
        ``ANTHROPIC_BASE_URL`` if set.
        """
        model = os.getenv("TEACHER_MODEL", DEFAULT_ANTHROPIC_MODEL)
        kwargs: dict[str, Any] = {"model": model}
        retries = os.getenv("TEACHER_MAX_RETRIES")
        if retries:
            kwargs["max_retries"] = int(retries)
        kwargs.update(overrides)
        return cls(**kwargs)

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Load .env so ANTHROPIC_API_KEY can live in the gitignored repo .env,
            # consistent with how OllamaTool resolves config.
            try:
                from dotenv import load_dotenv

                from local_agentic_analytics.core.config import PROJECT_ROOT

                load_dotenv(PROJECT_ROOT / ".env", override=False)
            except Exception:  # noqa: BLE001 - dotenv is best-effort
                pass

            import anthropic  # Imported lazily so offline tests need no key/SDK.

            self._client = anthropic.Anthropic(max_retries=self.max_retries)
        return self._client

    def generate(
        self, prompt: str, temperature: float = 0.4, max_tokens: int = 384
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        client = self._ensure_client()
        # temperature is intentionally dropped: Opus 4.8 rejects sampling params.
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("Teacher refused the request (stop_reason=refusal)")

        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise RuntimeError("Teacher response did not contain text")
        return text


# --- OpenAI-compatible teachers (DeepSeek, Gemini) -----------------------------

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
# 2.5-flash is the flash tier with free-tier quota; newer 3.x flash models
# currently require billing (429 quota) on a free key. Override via TEACHER_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Same-family teacher wave: pick the id explicitly via TEACHER_MODEL. The
# default only serves as a safe placeholder for the gemma branch.
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"


class OpenAICompatibleTeacherClient:
    """Teacher for any OpenAI-compatible ``/chat/completions`` endpoint.

    Used by non-Anthropic providers (DeepSeek, Gemini) via the lightweight
    ``requests`` path rather than a vendor SDK. Unlike Opus 4.8 these accept
    ``temperature``, so it is sent. Subclasses set provider-specific defaults;
    ``post`` is injectable so the request path is testable offline.
    """

    PROVIDER = "OpenAI-compatible"
    DEFAULT_BASE_URL = ""
    DEFAULT_MODEL = ""
    API_KEY_ENVS: tuple[str, ...] = ()
    BASE_URL_ENV = ""
    # Provider-specific request fields merged into every payload (e.g. Gemini
    # disables thinking so reasoning tokens don't consume the max_tokens budget).
    EXTRA_PAYLOAD: dict[str, Any] = {}

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: int = 180,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        max_backoff: float = 30.0,
        sleep: Any = time.sleep,
        post: Any = None,
        extra_payload: dict[str, Any] | None = None,
    ):
        model = model or self.DEFAULT_MODEL
        base_url = base_url or self.DEFAULT_BASE_URL
        if not api_key or not api_key.strip():
            raise ValueError("api_key must not be empty")
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if not base_url or not base_url.strip():
            raise ValueError("base_url must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self._sleep = sleep
        self._post = post  # injectable for offline tests
        self.extra_payload = (
            dict(self.EXTRA_PAYLOAD) if extra_payload is None else dict(extra_payload)
        )

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenAICompatibleTeacherClient":
        """Build from the provider's API-key env var(s) + ``TEACHER_MODEL``."""
        try:
            from dotenv import load_dotenv

            from local_agentic_analytics.core.config import PROJECT_ROOT

            load_dotenv(PROJECT_ROOT / ".env", override=False)
        except Exception:  # noqa: BLE001 - dotenv is best-effort
            pass

        api_key = ""
        for name in cls.API_KEY_ENVS:
            value = os.getenv(name, "")
            if value.strip():
                api_key = value
                break
        if not api_key.strip():
            names = " / ".join(cls.API_KEY_ENVS) or "API key"
            raise ValueError(f"{names} is not set (env or .env)")

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "model": os.getenv("TEACHER_MODEL", cls.DEFAULT_MODEL),
            "base_url": os.getenv(cls.BASE_URL_ENV, cls.DEFAULT_BASE_URL),
        }
        retries = os.getenv("TEACHER_MAX_RETRIES")
        if retries:
            kwargs["max_retries"] = int(retries)
        kwargs.update(overrides)
        return cls(**kwargs)

    def _backoff_seconds(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.max_backoff)
        return min(self.backoff_base * (2**attempt), self.max_backoff)

    def generate(
        self, prompt: str, temperature: float = 0.4, max_tokens: int = 384
    ) -> str:
        import requests

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        post = self._post or requests.post
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **self.extra_payload,
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = post(
                    url, json=payload, headers=headers, timeout=self.timeout_seconds
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self._sleep(self._backoff_seconds(attempt, None))
                    continue
                raise RuntimeError(f"{self.PROVIDER} request failed: {exc}") from exc

            status = getattr(response, "status_code", 200)
            if status in _RETRYABLE_STATUS:
                headers_obj = getattr(response, "headers", None) or {}
                retry_after = _parse_retry_after(headers_obj.get("Retry-After"))
                last_error = RuntimeError(f"{self.PROVIDER} endpoint returned {status}")
                if attempt < self.max_retries:
                    self._sleep(self._backoff_seconds(attempt, retry_after))
                    continue
                raise last_error
            if status >= 400:
                body = getattr(response, "text", "") or ""
                raise RuntimeError(f"{self.PROVIDER} API error {status}: {body[:300]}")

            data = response.json()
            text = _extract_chat_text(data)
            if not text:
                raise RuntimeError(f"{self.PROVIDER} response did not contain text")
            return text

        raise RuntimeError(f"{self.PROVIDER} request exhausted retries: {last_error}")


class DeepSeekTeacherClient(OpenAICompatibleTeacherClient):
    """Teacher backed by the DeepSeek API. Set the model via ``TEACHER_MODEL``
    (e.g. ``deepseek-v4-flash``) and the key via ``DEEPSEEK_API_KEY``."""

    PROVIDER = "DeepSeek"
    DEFAULT_BASE_URL = DEFAULT_DEEPSEEK_BASE_URL
    DEFAULT_MODEL = DEFAULT_DEEPSEEK_MODEL
    API_KEY_ENVS = ("DEEPSEEK_API_KEY",)
    BASE_URL_ENV = "DEEPSEEK_BASE_URL"


class GeminiTeacherClient(OpenAICompatibleTeacherClient):
    """Teacher backed by Gemini's OpenAI-compatible endpoint. Set the model via
    ``TEACHER_MODEL`` (e.g. ``gemini-2.5-flash``) and the key via
    ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)."""

    PROVIDER = "Gemini"
    DEFAULT_BASE_URL = DEFAULT_GEMINI_BASE_URL
    DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
    API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    BASE_URL_ENV = "GEMINI_BASE_URL"
    # 2.5 flash is a thinking model; disable thinking so the visible answer isn't
    # starved by reasoning tokens (which otherwise truncates the narration).
    EXTRA_PAYLOAD = {"reasoning_effort": "none"}


class OpenRouterTeacherClient(OpenAICompatibleTeacherClient):
    """Teacher backed by OpenRouter's OpenAI-compatible endpoint.

    Used for the same-family teacher wave (a Gemma teacher for the gemma
    student, a Qwen teacher for the qwen student). Set the model via
    ``TEACHER_MODEL`` and the key via ``OPENROUTER_API_KEY``, e.g.::

        TEACHER_MODEL=google/gemma-4-31b-it
        TEACHER_MODEL=Qwen/Qwen3.5-27B

    Two reproducibility notes:

    * Use the PAID model ids (no ``:free`` suffix). Free variants are capped at
      50 requests/day until 10+ credits have been purchased, which is far below
      one dataset wave.
    * OpenRouter may route the same model id to different upstream providers,
      which can differ in quantization. ``allow_fallbacks: False`` pins the
      route so a wave is not silently served by a different backend; set
      ``TEACHER_PROVIDER_ORDER`` (comma-separated) to also fix the provider.
      Record the resolved provider in the run manifest.

    Reasoning-style models (the Qwen 3.x line) think by default. Disable it via
    ``TEACHER_EXTRA_PAYLOAD`` so reasoning tokens do not eat the answer budget::

        TEACHER_EXTRA_PAYLOAD={"reasoning": {"enabled": false}}
    """

    PROVIDER = "OpenRouter"
    DEFAULT_BASE_URL = DEFAULT_OPENROUTER_BASE_URL
    DEFAULT_MODEL = DEFAULT_OPENROUTER_MODEL
    API_KEY_ENVS = ("OPENROUTER_API_KEY",)
    BASE_URL_ENV = "OPENROUTER_BASE_URL"
    EXTRA_PAYLOAD = {"provider": {"allow_fallbacks": False}}

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenRouterTeacherClient":
        """Build from env, honouring provider pinning and payload overrides."""
        client = super().from_env(**overrides)

        order = os.getenv("TEACHER_PROVIDER_ORDER", "").strip()
        if order:
            provider = dict(client.extra_payload.get("provider", {}))
            provider["order"] = [p.strip() for p in order.split(",") if p.strip()]
            client.extra_payload["provider"] = provider

        raw_extra = os.getenv("TEACHER_EXTRA_PAYLOAD", "").strip()
        if raw_extra:
            try:
                client.extra_payload.update(json.loads(raw_extra))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"TEACHER_EXTRA_PAYLOAD is not valid JSON: {exc}"
                ) from exc

        return client


def _extract_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""
