#!/usr/bin/env python3
"""
GSM8K client runner for vLLM v0.34.0 server (per-request metrics).

Prerequisite — a live vLLM server started from launcherFID01.

Reads FAULT_MODE, FAULT_P, FAULT_SEED from environment for provenance.
Writes outputs.jsonl, meta.json, server_metrics.txt into --out.
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
DEFAULT_BASE_URL = "http://localhost:8000/v1"

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step. "
    "End your response with a final line of the form '#### <number>'."
)
ANSWER_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--resume", action="store_true")
    # Recorded for provenance only:
    p.add_argument("--prefix-caching", choices=["on", "off"], default="on")
    p.add_argument("--tp", type=int, default=1, choices=[1, 2, 4, 8])
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=128)
    return p.parse_args()

def _run(cmd):
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def git_info():
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }

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
        gpus = [torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())]
    except Exception:
        gpus = []
    driver = None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
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

def _server_metrics_text(base_url):
    try:
        url = base_url.replace("/v1", "/metrics")
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"# unavailable: {e}\n"

def _percentile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]

def _get_field(m, name):
    if isinstance(m, dict):
        return m.get(name)
    return getattr(m, name, None)

def _extract_metrics(response):
    m = getattr(response, "metrics", None)
    if m is None:
        extra = getattr(response, "model_extra", None) or {}
        m = extra.get("metrics")
    if m is None:
        return {"ttft_ms": None, "queue_ms": None, "generation_ms": None,
                "itl_ms": None, "tokens_per_second": None}
    return {
        "ttft_ms":           _get_field(m, "time_to_first_token_ms"),
        "queue_ms":          _get_field(m, "queue_time_ms"),
        "generation_ms":     _get_field(m, "generation_time_ms"),
        "itl_ms":            _get_field(m, "mean_itl_ms"),
        "tokens_per_second": _get_field(m, "tokens_per_second"),
    }

def _extract_logprobs(choice):
    if not (getattr(choice, "logprobs", None) and choice.logprobs.content):
        return [], []
    tokens, lps = [], []
    for t in choice.logprobs.content:
        tokens.append(getattr(t, "token", None))
        lps.append(getattr(t, "logprob", None))
    return tokens, lps

def _load_done_idx(out_path, resume):
    done_idx = set()
    if resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if "error" not in row:
                    done_idx.add(row["idx"])
        print(f"[resume] {len(done_idx)} completed rows already in {out_path}")
        return done_idx, "a"
    if os.path.exists(out_path):
        sys.exit(f"Refusing to overwrite {os.path.dirname(out_path)}; use --resume")
    return done_idx, "w"

def _send_request(client, model, conv, max_tokens, seed):
    return client.chat.completions.create(
        model=model,
        messages=conv,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=True,
        top_logprobs=1,
        extra_body={"include_metrics": True, "seed": seed},
    )

def _build_success_row(idx, response, example, fault_mode):
    choice = response.choices[0]
    text = choice.message.content or ""
    tokens, lps = _extract_logprobs(choice)

    gold_label = parse_answer(example["answer"])
    prediction = parse_answer(text)
    correct = (gold_label is not None
               and prediction is not None
               and gold_label == prediction)
    is_length_stop = (choice.finish_reason == "length")

    usage = getattr(response, "usage", None)
    p_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    c_tokens = getattr(usage, "completion_tokens", None) if usage else None
    m = _extract_metrics(response)

    row = {
        "idx": idx,
        "request_id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
        "gold_label": gold_label,
        "prediction": prediction,
        "correct": correct,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "text": text,
        "tokens": tokens,
        "logprobs": lps,
        "ttft_ms": m["ttft_ms"],
        "queue_ms": m["queue_ms"],
        "generation_ms": m["generation_ms"],
        "itl_ms": m["itl_ms"],
        "tokens_per_second": m["tokens_per_second"],
        "fault_mode": fault_mode,
    }
    return row, correct, is_length_stop, p_tokens, c_tokens, m

def _build_error_row(idx, error, example):
    return {
        "idx": idx,
        "error": str(error),
        "gold_label": parse_answer(example["answer"]),
    }

def main():
    args = parse_args()

    out_path = os.path.join(args.out, "outputs.jsonl")
    os.makedirs(args.out, exist_ok=True)

    done_idx, write_mode = _load_done_idx(out_path, args.resume)

    from datasets import load_dataset
    from openai import OpenAI

    client = OpenAI(
        base_url=args.base_url, api_key="EMPTY", timeout=args.timeout,
    )

    try:
        client.models.list()
    except Exception as e:
        sys.exit(f"Cannot reach vLLM server at {args.base_url}: {e}")

    dataset = load_dataset("openai/gsm8k", "main")
    train, test = dataset["train"], dataset["test"]
    if args.limit is not None:
        test = test.select(range(min(args.limit, len(test))))

    remaining = len(test) - len(done_idx)
    if remaining <= 0:
        print(f"[{args.out}] nothing to do — all {len(test)} rows complete")
        return

    conversations = build_conversations(train, test, args.num_shots)

    fault_mode = os.environ.get("FAULT_MODE", "clean")
    fault_p = os.environ.get("FAULT_P")
    fault_seed = os.environ.get("FAULT_SEED")
    print(f"[fault] mode={fault_mode} p={fault_p} seed={fault_seed}")

    n_correct = 0
    n_ok = 0
    n_errors = 0
    n_length_stops = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    ttfts, queues, gens, itls = [], [], [], []

    t0 = time.time()
    with open(out_path, write_mode) as f:
        for i, (example, conv) in enumerate(zip(test, conversations)):
            if i in done_idx:
                continue

            try:
                response = _send_request(
                    client, args.model, conv, args.max_tokens, args.seed,
                )
            except Exception as e:
                n_errors += 1
                sys.stderr.write(f"[{i}] request failed: {e}\n")
                f.write(json.dumps(_build_error_row(i, e, example)) + "\n")
                f.flush()
                continue

            (row, correct, is_length_stop,
             p_tokens, c_tokens, m) = _build_success_row(
                i, response, example, fault_mode,
            )

            if correct:
                n_correct += 1
            if is_length_stop:
                n_length_stops += 1
            n_ok += 1

            if p_tokens:
                total_prompt_tokens += p_tokens
            if c_tokens:
                total_completion_tokens += c_tokens

            if m["ttft_ms"] is not None:       ttfts.append(m["ttft_ms"])
            if m["queue_ms"] is not None:      queues.append(m["queue_ms"])
            if m["generation_ms"] is not None: gens.append(m["generation_ms"])
            if m["itl_ms"] is not None:        itls.append(m["itl_ms"])

            f.write(json.dumps(row) + "\n")
            f.flush()

            processed = n_ok + n_errors
            if processed % 50 == 0:
                elapsed = time.time() - t0
                acc_live = (n_correct / n_ok) if n_ok else 0.0
                print(f"[{processed}/{remaining}] elapsed={elapsed:.0f}s "
                      f"acc={acc_live:.4f} ok={n_ok} errors={n_errors}")

    wall_s = time.time() - t0

    sm_path = os.path.join(args.out, "server_metrics.txt")
    with open(sm_path, "w") as f:
        f.write(_server_metrics_text(args.base_url))

    n_attempted = n_ok + n_errors
    accuracy_ok_only = (n_correct / n_ok) if n_ok else 0.0
    accuracy_with_errors = (n_correct / n_attempted) if n_attempted else 0.0

    from openai import __version__ as openai_ver

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "fault_condition": {"mode": fault_mode, "p": fault_p, "seed": fault_seed},
        "git": git_info(),
        "model_revision": _model_revision(args.model),
        "versions": {
            "openai_client": openai_ver,
            "python": platform.python_version(),
        },
        "hardware": _hardware(),
        "server_version": _server_version(args.base_url),
        "results": {
            "n_attempted": n_attempted,
            "n_ok": n_ok,
            "n_errors": n_errors,
            "n_correct": n_correct,
            "accuracy_ok_only": accuracy_ok_only,
            "accuracy_with_errors": accuracy_with_errors,
            "length_stops": n_length_stops,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "wall_s": wall_s,
            "tokens_per_sec_client": (total_completion_tokens / wall_s) if wall_s > 0 else None,
            "ttft_ms_p50": _percentile(ttfts, 50),
            "ttft_ms_p90": _percentile(ttfts, 90),
            "ttft_ms_p99": _percentile(ttfts, 99),
            "queue_ms_p50": _percentile(queues, 50),
            "queue_ms_p90": _percentile(queues, 90),
            "generation_ms_p50": _percentile(gens, 50),
            "generation_ms_p90": _percentile(gens, 90),
            "itl_ms_p50": _percentile(itls, 50),
            "itl_ms_p90": _percentile(itls, 90),
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    r = meta["results"]
    print(
        f"[{args.out}] fault={fault_mode} "
        f"attempted={n_attempted} ok={n_ok} errors={n_errors} "
        f"acc_ok={r['accuracy_ok_only']:.4f} "
        f"acc_all={r['accuracy_with_errors']:.4f} "
        f"len_stops={n_length_stops} wall={wall_s:.1f}s "
        f"ttft_p50={r['ttft_ms_p50']}ms ttft_p90={r['ttft_ms_p90']}ms "
        f"itl_p50={r['itl_ms_p50']}ms"
    )

if __name__ == "__main__":
    main()