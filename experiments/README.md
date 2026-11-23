`baseline_gsm8k.py`
The baseline script follows a client-server model. One process will run `vllm serve` as a stand-alone server (`tmux` session)
accepting HTTP requests and the Python script is an API client, sending one HTTP request per question to the server.
The server returns a JSON response containing the generated test, logprobs, and a metrics object with per-request timings.
