# inspect-neuron — blind neuron explanations and steering probes

`inspect-neuron` is a small research toolkit for explaining individual MLP neurons and then
testing those explanations *blindly*: one agent inspects a neuron and writes an explanation, and
another agent — seeing only an anonymized description — writes texts that it believes will fire
that neuron. If the explanation is right, the new texts fire the neuron more often than their
near-miss twins do. A final stage probes whether forcing the documented neuron changes what the
model says.

It is inspired by Nick Cammarata, *Neural Networks Are Already Clean* (Goodfire,
<https://already-clean.goodfire.com/>, retrieved 2026-10-27). The essay is linked and cited,
**not** reproduced here; see [Citation and attribution](#citation-and-attribution).

The Python distribution is `inspect-neuron`; it installs one console command, `nn`. The
reference agent workflow is written for DeepSeek Harness (DSH), but the `nn` command line is
independent of it. Without DSH you can run the same three stages by hand; see
[docs/workflow.md](docs/workflow.md#manual-equivalent).

It is meant for good-faith interpretability and AI-safety research; read
[Responsible use](#responsible-use) before running it.

## Responsible use

`inspect-neuron` is published for **interpretability and AI-safety research**: understanding what
individual neurons do, testing explanations of them blindly, and studying how activation steering
changes model behaviour. It is not a jailbreaking, content-generation or offensive-security
toolkit.

The pipeline deliberately searches for neurons that change model behaviour, and its steering
probes can make a model produce output that is harmful, biased, deceptive or offensive. Those
outputs are **research evidence about a model**, not advice, instructions or content to publish or
distribute. Do not use this repository to build or improve systems intended to manipulate people,
bypass safety measures, spread misinformation, or otherwise cause harm. Do not use it against
models, systems or people you are not authorized to test.

You are responsible for the models, datasets and hardware you use, for complying with their
licenses and terms, and for how you handle the outputs. If you publish results, include the model,
the exact recipe and the limitations, and obtain any ethics or institutional review your context
requires. The software is provided **as-is, with no warranty**, for research use; see
[LICENSE](LICENSE).

## The three stages

1. **Find neurons.** Either screen an intent corpus against a benign contrast
   (`nn screen`), pick candidates, and harvest a targeted store (`nn harvest --neurons`), or
   start from an existing store and let `nn select` draw a random sample of live neurons. You
   can also pass an explicit list of neuron ids.
2. **Blind test.** For each neuron: inspectors examine both poles with `nn top`, `nn run`,
   `nn steer`, `nn ablate` and `nn attrib`; if you use more than one inspector, a synthesizer
   merges their findings into one card (`nn card submit`); a separate anonymizer per pole
   removes the examples and ids (`nn card redact`); blind testers who see only the anonymized
   text write new examples, near misses and distant texts; one pooled `nn score` call judges
   them; an optional paper writer summarizes the result and publishes a neuron page.
3. **Steering probe.** For every scored neuron, one calibrated `nn generate --json --chat`
   call per position policy applies a fixed grid of `add`, `set` and `scale` doses. A judge
   labels every generation; the workflow deterministically selects the best coherent
   on-concept run (or records that there is none), and — when a judged candidate and a
   calibration exist — saves one showcase record per neuron with `nn steering save`. The report
   and HTML library show it.

The workflow files are `.dsh/workflows/blind-neuron-test.js` and
`.dsh/workflows/blind-neuron-test.meta.json`. The vocabulary and data layout are described in
[DESIGN_DOC.md](DESIGN_DOC.md); the agent-facing guide is [AGENTS.md](AGENTS.md). New to the
repository? Start with the
[new-session checklist](AGENTS.md#new-session--new-agent-start-here).

## Requirements

- **Python 3.10 or newer.** 3.10, 3.11 and 3.12 are tested.
- **A CUDA GPU with enough VRAM for the model plus the harvest arrays.** The model loads in
  bf16. `Qwen/Qwen3-0.6B` (the default example) needs roughly 4 GB of GPU memory including the
  store arrays and generation cache; larger models scale with their parameter count. CPU
  (`--device cpu`) works for the small model and for smoke tests but is slow.
- **Only gated-MLP architectures are supported.** The code assumes each decoder layer's MLP has
  `gate_proj`, `up_proj` and `down_proj` and computes `act_fn(gate_proj(x)) * up_proj(x)`.
- **Disk** for the model cache and the stores. Hugging Face downloads are cached in `HF_HOME`
  (default `~/.cache/huggingface`); point `HF_HOME` at a disk with room. A full Qwen3-0.6B store
  is about 0.9 GB even for a tiny corpus, because `vals.npy` and `windows.npy` are dense over
  every layer and neuron. A targeted store (`--neurons`) is about 13 MB for the same model,
  mostly the tokenizer copy and the dense `fires.npy` array.
- **Harvest memory is not only disk.** `nn harvest` keeps `vals.npy`, `windows.npy` and
  `fires.npy` on the GPU while it runs. A 27B model (64 layers × 17,408 neurons) at top-k 20 and
  window 32 needs about 11.8 GB of extra GPU memory on top of the weights, not just 11.8 GB of
  disk.
- **Optional:** an existing DeepSeek Harness (DSH) session to launch the agent workflow. Without
  it, the manual path in [docs/workflow.md](docs/workflow.md#manual-equivalent) is supported and
  complete.
- `nvidia-smi` if you want to check GPU use. GPUs may be shared; never kill, restart or attach
  to a process you did not start.

## Install

Dependencies are downloaded from PyPI, so the first install needs network access. Check the
interpreter first: Python 3.10 or newer is required. On macOS the system `python3` is often 3.9,
and its old bundled pip fails with a misleading `setup.py`/`setup.cfg not found` or
`editable mode requires setuptools` error before revealing the real `requires >=3.10` failure.
If the check fails, create the venv with a newer interpreter, for example
`python3.12 -m venv .venv`.

```bash
python3 --version   # must be 3.10 or newer; on macOS /usr/bin/python3 is often 3.9
git clone https://github.com/simveit/inspect-neuron.git inspect-neuron && cd inspect-neuron
python3 -m venv .venv   # or: python3.12 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/nn --help
./scripts/nn.sh --check
```

`pip install -e .` installs torch, transformers, datasets, accelerate, numpy and tokenizers. pip
may choose a CPU-only torch wheel; for CUDA, install the wheel matching your CUDA version from
<https://pytorch.org/get-started/locally/> first, then re-run the editable install.

The committed runner `scripts/nn.sh` prefers `.venv/bin/nn` and falls back to `nn` on `PATH`.
`./scripts/nn.sh --check` works locally and prints the `nn` binary, Python version, store status,
GPU information and free disk. It does not start or stop anything.

It runs **locally by default**. If a machine-local `run/remote.sh` exists (copy the templates in
[run.example/](run.example/) into a gitignored `run/`), the runner delegates every call — including
`--cmd` and `--detach` — to your GPU host. Local mode is the normal path for a newcomer; remote
mode is optional and documented in [docs/remote.md](docs/remote.md).

## Quickstart on your own dataset

The commands below assume the repository root and a Bash-like shell. Replace `YOUR/DATASET`,
`YOUR_GPU` and `/path/to/store` with your own values. Paths passed to the workflow should be
absolute. Neuron ids such as `L14.N957` are illustrative; use ids from your own store.

### 1. Choose a corpus and a store

`nn harvest` streams FineWeb (`sample-10BT`) by default. Any Hugging Face dataset or local
`.jsonl`/`.txt` file can replace it.

```bash
export NN_STORE="$PWD/runs/my-store"       # absolute; runs/ is gitignored
export NN_SERVER="http://127.0.0.1:8765"   # default anyway
export NN_RUN="r1"                         # a fresh label per experiment

# A Hugging Face dataset:
.venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e7 \
  --dataset YOUR/DATASET --split train --field text

# Or a local JSONL / plain-text file, one document per line:
.venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e7 \
  --texts /path/to/corpus.jsonl
# .txt works too:
.venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e7 \
  --texts /path/to/corpus.txt
```

For a Hugging Face dataset you can also pass `--config`, and `--where KEY=VALUE` (repeatable)
to keep only rows whose column matches a value case-insensitively, for example
`--where is_safe=false`. `--field` may name several comma-separated columns; they are joined
with newlines. `--max-docs N` caps a run, which is the cheapest smoke test. The corpus source is
recorded in `meta.json`, and a resume cannot silently switch corpora.

`--tokens` is a target, not a hard cap: sequence packing can overshoot it by up to
`batch × ctx` tokens, and a finite corpus simply stops when it is exhausted.

The [examples/tiny-corpus.jsonl](examples/tiny-corpus.jsonl) file (60 synthetic documents,
no third-party text) is for plumbing checks:

```bash
NN_STORE="$PWD/runs/tiny-store" .venv/bin/nn harvest --device cpu \
  --texts examples/tiny-corpus.jsonl --tokens 50000 --max-docs 20
```

Even this tiny corpus builds a **full** all-neuron store of about 0.9 GB for Qwen3-0.6B. For a
small store, use a targeted harvest (next section); it is about 13 MB for the same model.

### 2. Optional: find a targeted set of neurons

A full harvest is the simplest start. To keep a smaller store, screen an intent corpus against
a benign contrast first. Local files are the quickest route:

```bash
.venv/bin/nn screen --model Qwen/Qwen3-0.6B --intent violence \
  --texts intent.jsonl --benign-texts benign.jsonl \
  --out /tmp/screen.json --top-k 60 --max-texts 120 --tokens 100
```

`--texts` and `--benign-texts` take one document per line. Texts shorter than `--tokens` are
skipped, and the screen needs at least two usable texts per class. To screen a Hugging Face
dataset instead, replace the file flags with `--dataset YOUR/DATASET --split train --field text
--where label=harmful` and the matching `--benign-dataset`, `--benign-split`, `--benign-field`
and `--benign-where` flags.

Read `/tmp/screen.json`: each entry has a neuron id, Welch t, class means/maxima and peak tokens.
Put the ids you want into `/tmp/picks.json`, either a list or `{"neurons": [...]}`:

```bash
NN_STORE="$PWD/runs/targeted-store" .venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e7 \
  --dataset YOUR/DATASET --split train --field text --neurons /tmp/picks.json
```

Use `--texts /path/to/corpus.jsonl` instead of the dataset flags to harvest a local file.

`nn top` reports corpus peaks measured deep inside long documents; a short standalone text can
activate the neuron far less. The blind test scores short texts against a short-context max, so
verify a candidate with `nn run` on a short text and choose examples carefully.

`nn rank --out /path/to/rank ...` is an alternative that ranks neurons by firing statistics on a
corpus and writes `top.json` for `--neurons`, alongside `fires.npy`, `sums.npy`, `peaks.npy` and
`meta.json`. The workflow can do the screen itself when you give it a `screen` config (below).

### 3. Start the model server

The blind chain needs a running server. The server loads the model named by the store's
`meta.json`:

```bash
NN_STORE=/path/to/store .venv/bin/nn serve
```

Wait until it answers, then keep it running for the workflow:

```bash
curl -fsS http://127.0.0.1:8765/info
```

The workflow does not start or stop the server. On a shared machine, only stop the server you
started. In remote mode, `./scripts/nn.sh --detach serve` starts it on the GPU host and prints
the pid and log path.

### 4. Run the three stages

**With DSH (reference path).** The workflow is not launched by a command name. The **Lead** is
the lead agent of an existing DeepSeek Harness (DSH) session
(<https://github.com/deepseek-ai/deepseek-harness>). The Lead reads the two workflow files and
calls the `workflow` tool with `meta` (the meta JSON object), `script` (the `.js` body) and
`args` (your run config). From a DSH session, ask:

> Run the blind-neuron-test workflow with args `{ run: 'r1', store: '/path/to/store', select: { n: 3, seed: 0 } }`.

A minimal local run against an existing store uses `nn select` to draw neurons:

```js
{
  run: 'r1',
  store: '/path/to/store',
  select: { n: 3, seed: 0, top: 20 },
  runner: './scripts/nn.sh',
  gpu: 0,
}
```

To have the workflow screen for neurons and harvest a targeted store, replace `select` with:

```js
screen: {
  intent: 'violence',
  dataset: { id: 'YOUR/DATASET', split: 'train', field: 'text', where: ['label=harmful'] },
  benign:  { id: 'YOUR/DATASET', split: 'train', field: 'text', where: ['label=benign'] },
  picks: 3,
  harvest_tokens: 2e6,
},
```

The workflow's meta declares **10 stage names**: Screen (optional), Setup, Inspect, Synthesize
(only when `inspectors > 1`), Anonymize, Test, Score, Paper, Steer and Report. Only Screen,
Setup, Steer and Report call `phase()`, so the progress display has four phases and the other
stages run inside them. Each agent carries a run-prefixed `opts()` label naming the task (for
example `r1:inspect:L14.N957`, not the stage title), and the `nn steering save` recorder adds a
`showcase` label, so there are 11 label kinds for the 10 stages. A run with `select: { n: 3 }`
takes 10–60 minutes on a small model depending on generation speed. The probe alone is 2
calibrated calls per neuron (6 generate calls total for 3 neurons), each carrying 27 doses, for
54 generations per neuron.

**Without DSH (manual path).** The same stages are ordinary `nn` commands. This path is
supported and complete; the exact test JSON, the showcase record schema and a
blindness-preserving procedure are in
[docs/workflow.md](docs/workflow.md#manual-equivalent).

```bash
# 1. Find: screen + harvest (above), or draw from an existing store.
.venv/bin/nn select 3 --seed 0 --top 20 --json

# 2. Blind test, per neuron: inspect with nn top/run/steer/ablate/attrib, then
.venv/bin/nn card submit L14.N957 --core "..." --negative "..." --offshoots "..." --residue "..."
.venv/bin/nn card redact <card-id> --pole + --text "<abstract description, no examples or ids>"
#    A different person/model writes the tests from the redacted text alone.
.venv/bin/nn score /tmp/tests.json
.venv/bin/nn paper save L14.N957 --title "..." --summary "..." \
  --clean 4 --clean-why "..." --interesting 3 --interesting-why "..."

# 3. Probe, then report.
.venv/bin/nn generate L14.N957 --model Qwen/Qwen3-0.6B --chat --text "..." \
  --add 8,16,32,64,128,256,-8,-16,-32,-64,-128,-256 \
  --set 8,16,32,64,128,256,-8,-16,-32,-64,-128,-256 \
  --scale 1.5,2,3 --ref sigma --positions all --max-new-tokens 70 \
  --temperature 0.8 --seed 0 --json
#    Judge the JSON, then save one showcase record per neuron when a judged candidate and a
#    calibration exist:
.venv/bin/nn steering save --file /tmp/showcase.json
.venv/bin/nn library rebuild
mkdir -p reports/blind-r1
.venv/bin/nn report > reports/blind-r1/report.md
```

The point of the manual path is that the person or model writing the tests sees only
`nn card read <id> --pole <pole>`, never the card, the neuron id or the corpus.

### 5. Read the results

- `reports/blind-<run>/report.md` — the blind-test body and, when a steering record was saved,
  `## Showcase`. With the DSH
  workflow it also has `## Steering` with the calibration, the selection and a table of every
  judged run.
- `reports/blind-<run>/<neuron>.html` — exists only for a pole that **won** its blind test, and
  is copied there by the workflow when the page is stamped with this run. TIEs and LOSSes add
  no page (common on the smoke model); `report.md` is always written and records every scored
  pole.
- `<store parent>/neuron-library/<model>/<neuron>.html` and `index.html` — the browsable
  library, written by `nn paper save` and `nn library rebuild`.
- `reports/` and `runs/` are gitignored; do not commit experiment artifacts.

## Configuration

Every command reads the environment, and flags override it:

| Variable | Meaning | Default |
|---|---|---|
| `NN_STORE` | store directory (harvest output, cards, scores, papers, steering records) | required |
| `NN_SERVER` | URL of the running `nn serve` | `http://127.0.0.1:8765` |
| `NN_RUN` | run label stamped on records and used for lookups | none |
| `NN_LIBRARY` | neuron library root (HTML pages) | `neuron-library/` beside the store |
| `HF_HOME` | Hugging Face cache | the library default |
| `CUDA_VISIBLE_DEVICES` | GPU pin for harvest/screen/generate | all visible |

Global flags go **before** the subcommand: `nn --store PATH --run r1 report`. Flags after the
subcommand are command-specific. `./scripts/nn.sh` forwards the variables above to `nn` and, in
local mode, to `--cmd` commands. In remote mode it forwards them to the GPU host. Every flag is
listed by `nn --help` and `nn <command> --help`.

The workflow config is a JavaScript object. Its top-level keys are:

| Key | Meaning | Default |
|---|---|---|
| `run` | run label; use a fresh one | `main` |
| `store` | harvested store; **required** | none |
| `server` | `nn serve` URL as seen by the runner | `http://127.0.0.1:8765` |
| `runner` | command prefix for every `nn` call | `./scripts/nn.sh` |
| `neurons` | explicit list of ids; bypasses `select` | `[]` |
| `select` | `{ n, seed, top }` random live neurons; `top: null` = all | `{ n: 3, seed: 0, top: 20 }` |
| `screen` | `null`, or an intent-vs-benign object (see below) | `null` |
| `work` | blind-test JSON directory | `<store parent>/blind-<run>` |
| `reports` | report folder in this checkout | `reports` |
| `batch` | neurons in flight | `3` |
| `inspectors` | inspectors per neuron; a synthesizer runs when > 1 | `1` |
| `testers` | blind testers per pole | `3` |
| `paper` | write the paper and library page | `true` |
| `steer` | probe options; `{}` = the defaults | `{}` |
| `gpu` | `CUDA_VISIBLE_DEVICES` for screen/harvest/generate | `null` |

`screen` takes `intent`, `dataset`, `benign`, `picks`, `top_k`, `max_texts`, `tokens`,
`harvest_tokens`, `model` and `store`. `dataset` and `benign` accept `{ id, config, split,
field, texts, where, match, max_docs }`. Its targeted store is derived under the configured
store's parent (`<store parent>/store-<run>` unless `screen.store` is set); the workflow uses
that derived store for the blind chain and probe, not a path returned by the agent. Texts shorter
than `tokens` are skipped.

For `steer`, the defaults are sigma units, `add`/`set` ±8..256, `scale` 1.5/2/3, positions
`all` and `new`, 70 new tokens at temperature 0.8. For `add` and `set`, a positive strength
targets the `+` pole and a negative strength the `-` pole. `scale` is a plain multiplier and
preserves the sign of the activation already present, so the positive-targets-plus rule applies
to `add`/`set` only. At the CLI, empty mode grids are legal and produce calibration plus BASE
only: `nn generate ... --add '' --set '' --scale ''`.

Validation covers `inspectors`, `testers`, `batch`, `screen.picks`, `run`, `runner` and every
`steer` field. `select` and the remaining `screen` fields are passed through to `nn`; a bad value
fails there rather than before the run. Unknown `steer` keys throw, so a stale option fails
before any agent runs.

The full option reference, including all commands and their important flags, is
[docs/configuration.md](docs/configuration.md). The workflow launch protocol and the manual
equivalent are [docs/workflow.md](docs/workflow.md).

## Outputs

Inside the store:

```
meta.json  vals.npy  windows.npy  fires.npy  tokenizer.json  short_max.json
cards/       explanation cards (nn card submit)
scores/      per-test and pooled scores (nn score)
papers/      paper records and library metadata (nn paper save)
steering/    showcase records (nn steering save)
```

`short_max.json` is written by `nn score`; it caches each pole's short-context maximum.

In the store's parent directory:

```
blind-<run>/             blind-test JSON written by the workflow (work: null default)
neuron-library/          HTML pages, one per neuron with a winning pole, plus index.html
```

In this checkout (gitignored):

```
reports/blind-<run>/report.md + <neuron>.html
runs/                    pulled stores and local experiment data
run/                     machine-local runner config and logs (never committed)
```

`nn report` renders the newest records for `NN_RUN` (all runs without it). A run-scoped
`nn library rebuild` never deletes another run's page while that run still has a winning score:
a page is removed only when it belongs to the rebuilt run or no run wins the neuron anymore.
Pages without readable embedded metadata (hand-written files) are never deleted.

A showcase record is saved when the judge returns a candidate and a calibration exists; if not,
the report's `## Steering` section records that none was saved. `control` may be `null` when no
coherent control run exists. In a steering record, `run` and `model` are optional: `run`
defaults to `--run`/`NN_RUN`, and `model` defaults to the store's model.

## Verification

From a fresh checkout, without a GPU:

```bash
python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/nn-pyc" python3 -m py_compile nn/*.py tests/*.py examples/*.py
bash -n scripts/nn.sh
bash -n run.example/remote.sh.example
find . -name __pycache__ -type d -prune -exec rm -rf {} +
```

The suite has 48 dependency-free tests and needs only `numpy`. It covers pure helpers and score
math, the run-scoped library rule, showcase validation and rendering, the docs/workflow
structural contract, and a machine-leak guard. It has no subprocess tests: it never executes
`scripts/nn.sh` or loads a model. The manual smoke checklist is
[docs/manual-smoke.md](docs/manual-smoke.md); [docs/verification.md](docs/verification.md)
explains the static checks, the test suite and the CI configuration.

## Troubleshooting

| Symptom | Start here |
|---|---|
| `nn: command not found` | activate or call `.venv/bin/nn`; `./scripts/nn.sh` finds the venv |
| `no store` / `meta.json` missing | export `NN_STORE` or pass `--store`; finish the harvest first |
| server does not answer | `curl "$NN_SERVER/info"`; check the model in `meta.json`; start `nn serve` yourself |
| CUDA out of memory | lower harvest `--batch`/`--ctx`, lower generation `--max-new-tokens`; use a targeted store or a smaller model |
| Hugging Face auth / private dataset | `hf auth login` (or `huggingface-cli login`) or `HF_TOKEN`; accept the dataset terms |
| dataset column errors | check `--split`, `--field` and `--where`; `--field` must name string columns |
| workflow stops at Setup | the store and server models must match; pass the store absolute and the runner path exists |
| no library page | that neuron had no winning pole; ties/losses stay in the store, by design |
| remote runner fails | check `run/run.env` (`NN_REMOTE_HOST`, `NN_REMOTE_REPO`), SSH access and the remote venv |
| pages use plain system fonts | intentional: the templates are self-contained, with no CDN links |

The detailed symptom → diagnosis → fix list is [docs/troubleshooting.md](docs/troubleshooting.md).

## Limits

- There is no ground truth for what a neuron "means". A WIN is predictive evidence that the
  explanation fires the neuron more often than its near misses; it is not proof, and a TIE or
  LOSS is a real result.
- The judge reads a compact digest that cuts each generation at 300 characters; it can fetch a
  cut run in full before labelling, but movement is still a coarse 0–3 judgment. It is a filter,
  not a measurement.
- One query can be unrepresentative. Pass a query that matters to the concept you are probing.
- Working dose windows are narrow and non-monotonic. "No coherent on-concept dose" is a real
  result and is recorded as such, not treated as a missing run. The steering probe's `all` and
  `new` position policies can also be indistinguishable for late-layer neurons or short
  generations; that is a property of the probe, not a missing run.
- The blind test measures whether a description is *predictive*, not whether it is the best or
  the only explanation. There is no automated test of prompt quality or explanation quality:
  the blind test is the quality signal, and anonymization and judging are themselves LLM
  judgments.
- `Qwen/Qwen3-0.6B` is a smoke-test model. Results need not transfer to larger models.
- The agent workflow depends on model behaviour and on a running server. Re-running a stage can
  give different texts even with the same seed, because agents are not deterministic.
- A full harvest and a 27B model are large GPU/disk jobs. Start with a small model, a targeted
  store and few neurons; see [docs/configuration.md](docs/configuration.md#storage).

## Repository layout

```
nn/                          CLI, model/server, harvest, screen, rank, score, report, library
.dsh/workflows/              reference DSH workflow (.js) and its meta JSON
scripts/nn.sh                local-by-default runner; delegates to run/remote.sh when present
run.example/                 committed templates for the gitignored run/ folder
examples/                    tiny synthetic corpus, generator and a minimal workflow config
tests/                       dependency-free unittest suite
docs/                        workflow, configuration, remote, verification, smoke, troubleshooting, citation
AGENTS.md                    agent guide, including the new-session checklist
DESIGN_DOC.md                what the pipeline is and why it is built this way
LICENSE                      MIT license text
CITATION.cff                 machine-readable citation
```

## Citation and attribution

The idea and the blind-test protocol come from Nick Cammarata's essay:

> Nick Cammarata. *Neural Networks Are Already Clean*. Goodfire.
> <https://already-clean.goodfire.com/> (retrieved 2026-10-27)

This repository is an independent implementation and does **not** include the essay text or its
figures; cite and link the original. The default model (`Qwen/Qwen3-0.6B`), the streaming corpus
(`HuggingFaceFW/fineweb`, `sample-10BT`) and dataset examples such as
`PKU-Alignment/BeaverTails` are third-party artifacts under their own terms. See
[docs/citation.md](docs/citation.md) and [CITATION.cff](CITATION.cff) for details.

## Contact

Questions, bug reports and corrections: open an issue at
<https://github.com/simveit/inspect-neuron/issues>.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

Copyright © 2026 Simon Veitner.

Third-party models, datasets and other dependencies remain under their own licenses; the MIT
license covers only the code and documentation in this repository and does not relicense them.
