from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.finetune.teacher_client import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicTeacherClient,
    DeepSeekTeacherClient,
    FakeTeacherClient,
    GeminiTeacherClient,
)


class _Block:
    def __init__(self, type_: str, text: str = ""):
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeAnthropic:
    """Mimics the shape of ``anthropic.Anthropic`` used by the teacher client."""

    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_generate_drops_sampling_params_and_omits_thinking():
    sdk = _FakeAnthropic(_Response([_Block("text", "Narasi rata-rata 1,09 kW.")]))
    client = AnthropicTeacherClient(client=sdk, model="claude-opus-4-8")

    out = client.generate("a prompt", temperature=0.4, max_tokens=384)

    assert out == "Narasi rata-rata 1,09 kW."
    call = sdk.messages.calls[0]
    # Opus 4.8 rejects these — they must never be sent.
    assert "temperature" not in call
    assert "top_p" not in call
    assert "top_k" not in call
    assert "thinking" not in call
    assert call["model"] == "claude-opus-4-8"
    assert call["max_tokens"] == 384
    assert call["messages"] == [{"role": "user", "content": "a prompt"}]


def test_generate_skips_non_text_blocks():
    sdk = _FakeAnthropic(
        _Response([_Block("thinking", "internal"), _Block("text", "hasil")])
    )
    client = AnthropicTeacherClient(client=sdk)
    assert client.generate("p") == "hasil"


def test_generate_raises_on_refusal():
    sdk = _FakeAnthropic(_Response([], stop_reason="refusal"))
    client = AnthropicTeacherClient(client=sdk)
    with pytest.raises(RuntimeError, match="refus"):
        client.generate("p")


def test_generate_raises_on_empty_text():
    sdk = _FakeAnthropic(_Response([_Block("thinking", "only thinking")]))
    client = AnthropicTeacherClient(client=sdk)
    with pytest.raises(RuntimeError, match="did not contain text"):
        client.generate("p")


def test_from_env_defaults_to_opus_48(monkeypatch):
    monkeypatch.delenv("TEACHER_MODEL", raising=False)
    client = AnthropicTeacherClient.from_env()
    assert client.model == DEFAULT_ANTHROPIC_MODEL == "claude-opus-4-8"


def test_teacher_error_is_logged_as_rejection_not_crash():
    # A failing teacher must not abort the batch — build_dataset records it.
    from local_agentic_analytics.finetune.dataset_builder import build_dataset

    class BoomClient:
        def generate(self, prompt, temperature=0.4, max_tokens=384):
            raise RuntimeError("simulated API failure")

    result = build_dataset(BoomClient(), n=3, seed=1, concept_bank_text="")
    assert result.accepted == []
    assert len(result.rejected) == 3
    assert all(
        r.reasons[0].startswith("teacher_error:") for r in result.rejected
    )


def test_fake_client_still_used_for_offline_default():
    # Sanity: the offline fake remains the default path for tests.
    assert FakeTeacherClient(seed=1).seed == 1


# --- DeepSeek (OpenAI-compatible) ----------------------------------------------


class _ChatResponse:
    def __init__(self, status_code=200, content="", headers=None, text=""):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}
        self.text = text

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": self._content}}]}


class _FakePost:
    """Returns queued responses in order; records each call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._responses.pop(0)


def test_deepseek_sends_temperature_and_auth_and_extracts_text():
    post = _FakePost([_ChatResponse(content="Narasi rata-rata 1,09 kW.")])
    client = DeepSeekTeacherClient(api_key="sk-test", model="deepseek-chat", post=post)

    out = client.generate("a prompt", temperature=0.4, max_tokens=384)

    assert out == "Narasi rata-rata 1,09 kW."
    call = post.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["json"]["model"] == "deepseek-chat"
    assert call["json"]["temperature"] == 0.4  # DeepSeek accepts sampling params
    assert call["json"]["max_tokens"] == 384
    assert call["json"]["messages"] == [{"role": "user", "content": "a prompt"}]


def test_deepseek_retries_on_503_then_succeeds():
    post = _FakePost(
        [_ChatResponse(status_code=503), _ChatResponse(content="ok hasil 1,09 kW")]
    )
    client = DeepSeekTeacherClient(
        api_key="sk-test", post=post, sleep=lambda *_: None, max_retries=3
    )
    assert client.generate("p") == "ok hasil 1,09 kW"
    assert len(post.calls) == 2


def test_deepseek_raises_after_exhausting_retries():
    post = _FakePost([_ChatResponse(status_code=503) for _ in range(5)])
    client = DeepSeekTeacherClient(
        api_key="sk-test", post=post, sleep=lambda *_: None, max_retries=2
    )
    with pytest.raises(RuntimeError, match="503"):
        client.generate("p")
    assert len(post.calls) == 3  # initial + 2 retries


def test_deepseek_raises_on_client_error_without_retry():
    post = _FakePost([_ChatResponse(status_code=400, text="bad model")])
    client = DeepSeekTeacherClient(
        api_key="sk-test", post=post, sleep=lambda *_: None, max_retries=3
    )
    with pytest.raises(RuntimeError, match="400"):
        client.generate("p")
    assert len(post.calls) == 1  # 4xx is not retried


def test_deepseek_empty_key_raises():
    with pytest.raises(ValueError, match="api_key"):
        DeepSeekTeacherClient(api_key="")


def test_deepseek_from_env_reads_model_and_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    monkeypatch.setenv("TEACHER_MODEL", "deepseek-v4-flash")
    client = DeepSeekTeacherClient.from_env()
    assert client.api_key == "sk-env"
    assert client.model == "deepseek-v4-flash"
    assert client.base_url == "https://api.deepseek.com"


# --- Gemini (OpenAI-compatible) ------------------------------------------------


def test_gemini_uses_openai_compatible_endpoint_and_sends_temperature():
    post = _FakePost([_ChatResponse(content="Narasi tegangan 239 Volt.")])
    client = GeminiTeacherClient(api_key="g-key", model="gemini-2.5-flash", post=post)

    out = client.generate("p", temperature=0.4, max_tokens=384)

    assert out == "Narasi tegangan 239 Volt."
    call = post.calls[0]
    assert call["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert call["headers"]["Authorization"] == "Bearer g-key"
    assert call["json"]["model"] == "gemini-2.5-flash"
    assert call["json"]["temperature"] == 0.4
    # Thinking disabled so reasoning tokens don't starve the visible answer.
    assert call["json"]["reasoning_effort"] == "none"


def test_deepseek_does_not_send_reasoning_effort():
    post = _FakePost([_ChatResponse(content="ok 1,09 kW")])
    client = DeepSeekTeacherClient(api_key="sk", model="deepseek-chat", post=post)
    client.generate("p")
    assert "reasoning_effort" not in post.calls[0]["json"]


def test_gemini_from_env_prefers_gemini_key_then_google_key(monkeypatch):
    # Stub out .env loading so the test only sees the injected env vars.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TEACHER_MODEL", raising=False)

    monkeypatch.setenv("GOOGLE_API_KEY", "google-fallback")
    client = GeminiTeacherClient.from_env()
    assert client.api_key == "google-fallback"
    assert client.model == "gemini-2.5-flash"

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-primary")
    client = GeminiTeacherClient.from_env()
    assert client.api_key == "gemini-primary"


def test_gemini_error_message_uses_provider_name():
    post = _FakePost([_ChatResponse(status_code=400, text="bad model")])
    client = GeminiTeacherClient(api_key="g", post=post, sleep=lambda *_: None)
    with pytest.raises(RuntimeError, match="Gemini API error 400"):
        client.generate("p")
