from pathlib import Path
import sys

import pytest
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_agentic_analytics.tools import ollama_tool
from local_agentic_analytics.tools.ollama_tool import OllamaTool


class FakeResponse:
    def __init__(self, payload=None, ok=True, status_code=200):
        self._payload = payload or {}
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_check_connection_returns_true_when_ollama_responds(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://localhost:11434/api/tags"
        assert timeout == 5
        return FakeResponse(ok=True)

    monkeypatch.setattr(requests, "get", fake_get)
    tool = OllamaTool(base_url="http://localhost:11434", model="qwen2.5:3b")

    assert tool.check_connection() is True


def test_check_connection_returns_false_when_ollama_is_down(monkeypatch):
    def fake_get(url, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", fake_get)
    tool = OllamaTool(base_url="http://localhost:11434", model="qwen2.5:3b")

    assert tool.check_connection() is False


def test_generate_posts_to_ollama_api(monkeypatch):
    captured_payload = {}

    def fake_post(url, json, timeout):
        captured_payload.update(json)
        assert url == "http://localhost:11434/api/generate"
        assert timeout == 120
        return FakeResponse(payload={"response": "  hello  "})

    monkeypatch.setattr(requests, "post", fake_post)
    tool = OllamaTool(
        base_url="http://localhost:11434/",
        model="qwen2.5:3b",
        context_window=1024,
        num_gpu=0,
    )

    result = tool.generate("Say hello", temperature=0.2, max_tokens=32)

    assert result == "hello"
    assert captured_payload["model"] == "qwen2.5:3b"
    assert captured_payload["stream"] is False
    assert captured_payload["keep_alive"] == "30m"
    assert captured_payload["options"] == {
        "temperature": 0.2,
        "num_predict": 32,
        "num_ctx": 1024,
        "num_gpu": 0,
    }


def test_generate_stores_ollama_profiling_metrics(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse(
            payload={
                "response": "  SELECT 1;  ",
                "total_duration": 2_000_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 42,
                "prompt_eval_duration": 250_000_000,
                "eval_count": 8,
                "eval_duration": 1_250_000_000,
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    tool = OllamaTool(base_url="http://localhost:11434", model="qwen2.5:3b")

    assert tool.generate("SELECT query") == "SELECT 1;"
    assert tool.get_last_metrics() == {
        "total_duration": 2.0,
        "load_duration": 0.5,
        "prompt_eval_count": 42,
        "prompt_eval_duration": 0.25,
        "eval_count": 8,
        "eval_duration": 1.25,
    }


def test_generate_raises_when_ollama_is_not_running(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", fake_post)
    tool = OllamaTool(base_url="http://localhost:11434", model="qwen2.5:3b")

    with pytest.raises(RuntimeError, match="Ollama is not reachable"):
        tool.generate("Hello")


def test_from_config_reads_model_yaml_and_env_values(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(
        ollama_tool,
        "load_config",
        lambda path: {
            "model": {
                "provider": "ollama",
                "base_url": "${OLLAMA_BASE_URL}",
                "name": "${OLLAMA_MODEL}",
                "timeout_seconds": 90,
                "context_window": 2048,
                "num_gpu": 0,
                "keep_alive": "1h",
            }
        },
    )

    tool = OllamaTool.from_config()

    assert tool.base_url == "http://127.0.0.1:11434"
    assert tool.model == "qwen2.5:3b"
    assert tool.timeout_seconds == 90
    assert tool.context_window == 2048
    assert tool.num_gpu == 0
    assert tool.keep_alive == "1h"


def test_ollama_tool_validates_keep_alive_duration():
    assert (
        OllamaTool(
            base_url="http://localhost:11434",
            model="qwen2.5:3b",
            keep_alive="",
        ).keep_alive
        == "30m"
    )

    with pytest.raises(ValueError, match="keep_alive must be a duration string"):
        OllamaTool(
            base_url="http://localhost:11434",
            model="qwen2.5:3b",
            keep_alive="forever",
        )
