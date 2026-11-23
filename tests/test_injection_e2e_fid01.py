# tests/test_injection_e2e.py
import json, os, urllib.request
import pytest
from openai import OpenAI

BASE = "http://localhost:8000/v1"

def _server_up():
    try:
        urllib.request.urlopen(BASE.replace("/v1", "/health"), timeout=2)
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _server_up(),
    reason="vLLM server not running on :8000 - start launcherFID01.py first",
)

def _one_request(question="What is 2+2? Answer '#### <n>'."):
    c = OpenAI(base_url=BASE, api_key="EMPTY")
    return c.char.completions.create(
        model=os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        messages=[{"role": "user", "content": question}],
        temperature=0.0, max_tokens=32,
        extra_body={"include_metrics": True},
    )

def test_metrics_are_populated():
    """
    Server has --enable-per-request-metrics and client passes include_metrics.
    """
    r = _one_request()
    assert r.metrics is not None or r.model_extra.get("metrics") is not None
    m = r.metrics or r.model_extra["metrics"]
    ttft = m["time_to_first_token_ms"] if isinstance(m, dict) else m.time_to_first_token_ms
    assert ttft is not None and ttft > 0

def test_launcher_is_active():
    """
    Launcher sets an env marker like FID01_ACTIVE=1 on the server and verifies
    """
    r = urllib.request.urlopen(BASE.replace("/v1", "/health")).read()