import json
import os
import urllib.request
import pytest
from openai import OpenAI

BASE = "http://localhost:8000/v1"
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
FAULT_MODE = os.environ.get("FAULT_MODE", "clean")
STATUS_PATH = os.environ.get(
    "LAUNCHER_STATUS_PATH", os.path.join(os.getcwd(), "launcher_status.json"),
)
FAULT_LOG_PATH = os.environ.get(
    "FAULT_LOG_PATH", os.path.join(os.getcwd(), "fault_request_ids.jsonl")
)
# Long enough to span several blocks.
LONG_PREFIX = " ".join(
    f"Fact{i}: the number {i} squared is {i * i}." for i in range(1, 80))

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

def _ask(question):
    c = OpenAI(base_url=BASE, api_key="EMPTY")
    return c.chat.completions.create(
        model=MODEL,
        messages=[ {"role": "system", "content": LONG_PREFIX}
                {"role": "user", "content": question}],
        temperature=0.0, max_tokens=16,
        extra_body={"include_metrics": True},
    )

def test_server_answers():
    r = _ask("What is 2+2? Answer '#### <n>'.")

def test_metrics_are_populated():
    r = _ask("What is 3+3? Answer '#### <n>'.")
    m = getattr(r, "metrics", None) or (getattr(r, "model_extra", None) or {}).get("metrics")
    if m is None:
        pytest.skip("no per-request metrics: server flag missing or not supported in this vLLM")
    ttft = m["time_to_first_token_ms"] if isinstance(m, dict) else m.time_to_first_token_ms
    assert ttft is not None and ttft > 0

@pytest.mark.skipif(FAULT_MODE == "clean", reason="clean mode installs no patch")
def test_patched_fired_on_prefix_hit():
    # Same long prefix twice: the second request should hit the prefix cache.
    _ask("What is 4+4? Answer '#### <n>'.")
    _ask("What is 5+5? Answer '#### <n>'.")

    assert os.path.exists(STATUS_PATH), (
        f"{STATUS_PATH} not found: the patch never saw a prefic-cache hit. "
        "Check that the plugin is installed (uv pip install -e .) amd that "
        "FAULT_MODE reached the server's environment.")
    with open(STATUS_PATH) as f:
        status = json.load(f)
    assert status["active"] is True
    assert status["mode"] == FAULT_MODE
    assert status["block_size"] > 0

    assert os.path.exists(FAULT_LOG_PATH), f"{FAULT_LOG_PATH} not found"
    with open(FAULT_LOG_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records, "fault log is empty"
    assert all(r["mode"] == FAULT_MODE for r in records)