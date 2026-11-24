import json
import os
import urllib.request
import pytest
from openai import OpenAI

BASE = "http://localhost:8000/v1"
STATUS_PATH = os.environ.get(
    "LAUNCHER_STATUS_PATH",
    os.path.join(os.getcwd(), "launcher_status.json"),
)


def _server_up():
    try:
        urllib.request.urlopen(BASE.replace("/v1", "/health"), timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(),
    reason="vLLM server not running on :8000 — start launcherFID01 first",
)


def _one_request(question="What is 2+2? Answer '#### <n>'."):
    c = OpenAI(base_url=BASE, api_key="EMPTY")
    return c.chat.completions.create(
        model=os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        messages=[{"role": "user", "content": question}],
        temperature=0.0, max_tokens=32,
        extra_body={"include_metrics": True},
    )


def test_metrics_are_populated():
    r = _one_request()
    m = getattr(r, "metrics", None) or \
        (getattr(r, "model_extra", None) or {}).get("metrics")
    assert m is not None, "server not started with --enable-per-request-metrics"
    ttft = m["time_to_first_token_ms"] if isinstance(m, dict) \
           else m.time_to_first_token_ms
    assert ttft is not None and ttft > 0


def test_launcher_is_active():
    assert os.path.exists(STATUS_PATH), (
        f"{STATUS_PATH} not found — launcher was not started"
    )
    with open(STATUS_PATH) as f:
        status = json.load(f)
    assert status["active"] is True, f"launcher reports active=False: {status}"
    assert status["mode"] in {"over_aligned", "under_aligned", "misaligned"}