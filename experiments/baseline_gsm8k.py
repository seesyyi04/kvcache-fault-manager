#!/usr/bin/env python3
"""
GSM8K baseline runner for KV-cache fault experiments (vLLM v0.34 server API).

Prereq: launch the server once, e.g.
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 8 --enable-per-request-metrics --host 0.0.0.0 --port 8000

One run = one client sweep against that server. Writes into --out:
  outputs.jsonl: one line per question (text, logprobs, gold/pred, per-request metrics)
  meta.json: run metadata (args, versions, GPU, model revision, git commit, results)
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_URL = "http://localhost:8000/v1"
SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step. "
    "End your response with a final line of the form '#### <number>'."
)
ANSWER_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", required=True, help="output directory for this run")
    # The following flags are NOT consumed by this script — they are the
    # engine/server config. Recorded in meta.json for provenance only.
    p.add_argument("--prefix-caching", choices=["on", "off"], default="on")
    p.add_argument("--tp", type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--num-gpu-blocks-override", type=int, default=None)
    # Client-side
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def _run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
		return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def git_info():
    return {"commit": _run_cmd(["git", "rev-parse", "HEAD"]),
            "dirty": bool(_run_cmd(["git", "status", "--porcelain"]))}

def parse_answer(text):
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if m:
        s = m.group(1)
    else:
        matches = NUMBER_RE.findall(text)
        if not matches:
            return None
        s = matches[-1]
    s = s.replace(",", "").rstrip(".")
    if not s or s in {"-", "."}:
        return None
    try:
        return str(float(s))
    except ValueError:
        return None

def build_conversations(train, test, num_shots):
    shots = train.select(range(min(num_shots, len(train))))
    prefix = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots:
        prefix.append({"role": "user", "content": shot["question"]})
        prefix.append({"role": "assistant", "content": shot["answer"]})
    return [prefix + [{"role": "user", "content": ex["question"]}] for ex in test]

def _model_revision(model_name):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_name).sha
    except Exception as e:
        return f"unavailable: {e}"

def _hardware():
    try:
        import torch
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        gpus = []
    driver = None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            driver = r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return {"gpu_names": gpus, "nvidia_driver": driver}

def _server_version(base_url):
    try:
        url = base_url.replace("/v1", "/version")
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def main():
    args = parse_args()

    out_path = os.path.join(args.out, "outputs.jsonl")
    if os.path.exists(out_path):
        sys.exit(f"Refusing to overwrite run in {args.out}")
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset
    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

    dataset = load_dataset("openai/gsm8k", "main")
    train, test = dataset["train"], dataset["test"]
    if args.limit is not None:
        test = test.select(range(min(args.limit, len(test))))

    conversations = build_conversations(train, test, args.num_shots)

    n_correct = n_length_stops = total_out_tokens = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, (example, conv) in enumerate(zip(test, conversations)):
            response = client.chat.completions.create(
                model=args.model,
                messages=conv,
                max_tokens=args.max_tokens,
                temperature=0.0,
                seed=args.seed,
                logprobs=True,
                top_logprobs=1,
                include_metrics=True,
            )

            choice = response.choices[0]
            text = choice.message.content or ""

            if choice.logprobs and choice.logprobs.content:
                chosen = [t.logprob for t in choice.logprobs.content]
            else:
                chosen = [None] * len(text)

            gold_label = parse_answer(example["answer"])
            prediction = parse_answer(text)
            correct = (gold_label is not None
                       and prediction is not None
                       and gold_label == prediction)
            if correct:
                n_correct += 1
            if choice.finish_reason == "length":
                n_length_stops += 1

            m = getattr(response, "metrics", None)
            ttft_ms = getattr(m, "time_to_first_token_ms", None) if m else None
            queue_ms = getattr(m, "queue_time_ms", None) if m else None
            gen_ms = getattr(m, "generation_time_ms", None) if m else None
            itl_ms = getattr(m, "mean_itl_ms", None) if m else None
            tps = getattr(m, "tokens_per_second", None) if m else None

            n_out = len(chosen) if chosen else 0
            total_out_tokens += n_out

            f.write(json.dumps({
                "idx": i,
                "request_id": getattr(response, "id", None),
                "gold_label": gold_label,
                "prediction": prediction,
                "correct": correct,
                "finish_reason": choice.finish_reason,
                "text": text,
                "logprobs": chosen,
                "ttft_ms": ttft_ms,
                "queue_ms": queue_ms,
                "generation_ms": gen_ms,
                "itl_ms": itl_ms,
                "tokens_per_second": tps,
            }) + "\n")
    gen_s = time.time() - t0

    n = len(test)
    accuracy = (n_correct / n) if n else 0.0

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "git": git_info(),
        "model_revision": _model_revision(args.model),
        "versions": {
            "openai_client": getattr(__import__("openai"), "__version__", None),
            "python": platform.python_version(),
        },
        "hardware": _hardware(),
        "server_version": _server_version(BASE_URL),
        "results": {
            "n": n,
            "accuracy": accuracy,
            "length_stops": n_length_stops,
            "total_out_tokens": total_out_tokens,
            "client_wall_s": gen_s,
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{args.out}] n={n} acc={accuracy:.4f} length_stops={n_length_stops} "
          f"out_tokens={total_out_tokens} wall={gen_s:.1f}s")

if __name__ == "__main__":
    main()