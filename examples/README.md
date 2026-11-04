# Examples

- `tiny-corpus.jsonl` — 60 synthetic documents, written for this repository; no third-party
  text. Use it to check the plumbing, not to draw scientific conclusions.
- `make_tiny_corpus.py` — regenerates `tiny-corpus.jsonl` deterministically.
- `workflow-config.example.json` — a minimal DSH `workflow` args object. Replace
  `/path/to/store` with your store; see [../docs/workflow.md](../docs/workflow.md).

Smoke-sized harvest with the sample corpus:

```bash
export NN_STORE="$PWD/runs/tiny-store"
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/nn harvest --model Qwen/Qwen3-0.6B --texts examples/tiny-corpus.jsonl \
  --tokens 50000 --max-docs 20 --device cpu
.venv/bin/nn top L14.N957 --pole + -n 5   # L14.N957 is illustrative; use nn select
```

`nn harvest` streams the file and stops at its end, so `--tokens` is a target rather than a hard
cap; packing can overshoot it by up to `batch × ctx`. Even this tiny corpus builds a **full**
all-neuron store of about 0.9 GB for Qwen3-0.6B, because `vals.npy` and `windows.npy` are dense
over every layer and neuron. For a small store, screen or rank first and harvest with
`--neurons`; a targeted store is about 13 MB for the same model. The model download is the slow
part; use a GPU (`--device cuda`) for anything larger.
