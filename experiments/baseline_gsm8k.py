#!/usr/bin/env python3
"""
GSM8K baseline runner for KV-cache fault experiments (vLLM offline API).

One run = one engine configuration. Writes into --out:
  outputs.jsonl  one line per question (token IDs, logprobs, text, answer, ...)
  meta.json      run metadata (args, versions, GPU, model revision, git commit, results)
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = (
	"You are a careful math tutor. Solve the problem step by step. "
	"End your response with a final line of the form '#### <number>'."
)
ANSWER_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")   # GSM8K's final-answer format
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")            # fallback: last number in the text

def parse_args():
	p = argparse.ArgumentParser(description=__doc__,
								formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--model", default=DEFAULT_MODEL)
	p.add_argument("--out", required=True, help="output directory for this run")
	p.add_argument("--prefix-caching", choices=["on", "off"], default="on")
	p.add_argument("--num-gpu-blocks-override", type=int, default=None)
	p.add_argument("--tp", type=int, default=1, choices=[1, 2, 4, 8],
				   help="tensor parallel size (number of GPUs)")
	p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
	p.add_argument("--max-model-len", type=int, default=4096)
	p.add_argument("--max-num-seqs", type=int, default=128)
	p.add_argument("--max-tokens", type=int, default=512)
	p.add_argument("--num-shots", type=int, default=8)
	p.add_argument("--limit", type=int, default=None)
	p.add_argument("--seed", type=int, default=0)
	return p.parse_args()

def git_info():
	"""
	Commit hash of this repo + whether there are uncommitted changes.
	"""
	def run(cmd):
		try:
			return subprocess.run(cmd, capture_output=True, text=True,
								  cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
		except Exception:
			return None
	return {"commit": run(["git", "rev-parse", "HEAD"]),
			"dirty": bool(run(["git", "status", "--porcelain"]))}

def parse_answer(text):
	"""
	Extract the final numeric answer. Normalises (strips commas, trailing dot)
	and returns a canonical string (str(float(...))) so gold and pred compare
	with plain ==. Returns None if no number can be found.
	"""
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
	"""
	Same few-shot prefix (system + K user/assistant pairs from train) shared
	by every test question.
	"""
	shots = train.select(range(min(num_shots, len(train))))
	prefix = [{"role": "system", "content": SYSTEM_PROMPT}]
	for shot in shots:
		prefix.append({"role": "user", "content": shot["question"]})
		prefix.append({"role": "assistant", "content": shot["answer"]})
	return [prefix + [{"role": "user", "content": ex["question"]}] for ex in test]

def _num_cached_tokens(result):
	"""
	vLLM exposes this either on RequestOutput or on results.metrics.
	"""
	for obj in (result, getattr(result, "metrics", None)):
		if obj is not None and getattr(obj, "num_cached_tokens", None) is not None:
			return obj.num_cached_tokens
	return None

def _model_revision(llm):
	try:
		cfg = getattr(getattr(llm, "llm_engine", None), "model_config", None) \
			  or getattr(llm, "model_config", None)
		return getattr(cfg, "revision", None) if cfg is not None else None
	except Exception:
		return None

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
			capture_output=True, text=True, timeout=5,
		)
		if r.returncode == 0 and r.stdout.strip():
			driver = r.stdout.strip().splitlines()[0]
	except Exception:
		pass
	return {"gpu_names": gpus, "nvidia_driver": driver}

def _engine_metrics(llm):
	try:
		return [str(m) for m in llm.get_metrics()]
	except Exception as e:
		return [f"<unavailable: {e}>"]

def main():
	args = parse_args()

	out_path = os.path.join(args.out, "output.jsonl")
	if os.path.exists(out_path):
		sys.exit(f"Refusing to overwrite run in {args.out}")
	os.makedirs(args.out, exist_ok=True)

	# Heavy imports go here
	import torch
	import vllm
	from datasets import load_dataset
	from vllm import LLM, SamplingParams

	# Load data
	dataset = load_dataset("openai/gsm8k", "main")
	train, test = dataset["train"], dataset["test"]
	if args.limit is not None:
		test = test.select(range(min(args.limit, len(test))))

	# Build prompts
	conversations = build_conversations(train, test, args.num_shots)

	# Create the engine
	# '**' unpacks a dict into keyword arguments.
	engine_kwargs = dict(
		model = args.model,
		tensor_parallel_size = args.tp,
		enable_prefix_caching = (args.prefix_caching == "on"),
		gpu_memory_utilization = args.gpu_memory_utilization,
		max_model_len = args.max_model_len,
		max_num_seqs = args.max_num_seqs,
		seed = args.seed,
		disable_log_stats = False,
	)
	if args.num_gpu_blocks_override is not None:
		engine_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override

	t0 = time.time()
	llm = LLM(**engine_kwargs)
	load_s = time.time() - t0

	# Sampling settings
	# temperature=0 makes decoding greedy: always pick the likely token.
	#	Need this determinism to tell a fault apart from randomness.
	# logprobs=1 to record per-token logprob.
	# max_tokens caps rambling
	sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed, logprobs=1)

	# Generate
	t0 = time.time()
	outputs = llm.chat(conversations, sampling, use_tqdm=True)
	gen_s = time.time() - t0

	# Write results
	n_correct = n_length_stops = total_out_tokens = 0
	with open(out_path, "w") as f:
		for i, (example, result) in enumerate(zip(test, outputs)):
			completion = result.outputs[0]
			token_ids = list(completion.token_ids)
			# Per-position logprob of the token that was actually chosen:
			# completion.logprobs[pos] is {token_id: Logprob}; pick out token.
			if completion.logprobs is not None:
				chosen = []
				for pos, tok_id in enumerate(token_ids):
					lp = completion.logprobs[pos].get(tok_id)
					chosen.append(lp.logprob if lp is not None else None)
			else:
				chosen = [None] * len(token_ids)
	
			gold = parse_answer(example["answer"])
			pred = parse_answer(completion.text)
			correct = gold is not None and pred is not None and gold == pred
			if correct:
				n_correct += 1
			if completion.finish_reason == "length":
				n_length_stops += 1
			total_out_tokens += len(token_ids)

			f.write(json.dumps({
				"idx": i,
				"request_id": result.request_id,
				"gold": gold,
				"pred": pred,
				"correct": correct,
				"finish_reason": completion.finish_reason,
				"num_prompt_tokens": len(result.prompt_token_ids) if result.prompt_token_ids else 0,
				"num_cached_tokens": _num_cached_tokens(result),
				"text": completion.text,
				"token_ids": token_ids,
				"logprobs": chosen,
			}) + "\n")
	
	n = len(test)
	accuracy = (n_correct / n) if n else 0.0
	tok_per_s = (total_out_tokens / gen_s) if gen_s > 0 else 0.0

	# Metadata
	meta = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"args": vars(args),
		"engine_kwargs": engine_kwargs,
		"git": git_info(),
		"model_revision": _model_revision(llm),
		"versions": {
			"vllm": vllm.__version__,
			"torch": torch.__version__,
			"cuda": torch.version.cuda,
			"python": platform.python_version(),
		},
		"hardware": _hardware(),
		"results": {
			"n": n,
			"accuracy": accuracy,
			"length_stops": n_length_stops,
			"total_out_tokens": total_out_tokens,
			"load_s": load_s,
			"gen_s": gen_s,
			"tokens_per_sec": tok_per_s,
		},
		"engine_metrics": _engine_metrics(llm),
	}
	with open(os.path.join(args.out, "meta.json"), "w") as f:
		json.dump(meta, f, indent=2)

	print(
		f"[{args.out}] n={n} acc={accuracy:.4f} length_stops={n_length_stops} "
		f"out_tokens={total_out_tokens} load={load_s:.1f}s gen={gen_s:.1f}s "
		f"tok/s={tok_per_s:.1f}"
	)

if __name__ == "__main__":
	main()