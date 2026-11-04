# Configuration reference

This page lists the environment variables, the workflow config, every `nn` command and the
storage notes for `inspect-neuron`. `nn --help` and `nn <command> --help` are authoritative for
flags; this page explains what they mean and which defaults matter. Neuron ids in examples (for
example `L14.N957`) are illustrative.

## Environment

| Variable | Used by | Meaning | Default |
|---|---|---|---|
| `NN_STORE` | every store command | store directory: harvest output, cards, scores, papers, steering records | none (required) |
| `NN_SERVER` | commands that call the model API | URL of the running `nn serve` | `http://127.0.0.1:8765` |
| `NN_RUN` | record-writing and run-scoped reads | run label stamped on new cards, scores, papers and steering records | none |
| `NN_LIBRARY` | `nn paper save`, `nn library rebuild` | root of the HTML neuron library | `neuron-library/` beside the store |
| `HF_HOME` | Hugging Face downloads | model and dataset cache | the library default |
| `CUDA_VISIBLE_DEVICES` | harvest, screen, rank, serve, generate | which GPU(s) are visible | all visible devices |
| `NN_RUNNER` | `scripts/nn.sh` | explicit runner command; overrides `run/remote.sh` and local mode | unset |
| `NN_LOCAL` | `scripts/nn.sh` | `1` ignores a machine-local `run/remote.sh` and runs locally | unset |
| `NN_BIN` | `scripts/nn.sh` | explicit `nn` executable for local runs | `.venv/bin/nn`, else `nn` on `PATH` |

Flags override the environment: `--store`, `--server`, `--run` and `--library` exist on the
relevant commands. Global flags go **before** the subcommand:
`nn --store PATH --run r1 report`; flags after the subcommand are command-specific. `nn` sets
`OPENBLAS_NUM_THREADS=1` internally so importing numpy does not spawn a thread per core.

## Runner: local by default

`./scripts/nn.sh` is the one runner the workflow and the docs use:

```bash
./scripts/nn.sh --help                  # nn here (prefers .venv/bin/nn, else PATH)
./scripts/nn.sh --check                 # local: nn, python, store, GPU, disk
./scripts/nn.sh top L14.N957 --pole +    # any nn command
./scripts/nn.sh --cmd 'cat "$NN_STORE/meta.json"'   # a shell command in the same environment
./scripts/nn.sh --dry-run top L14.N957 --pole +     # print the command, run nothing
./scripts/nn.sh --detach serve          # background; prints pid and log path
NN_RUNNER=./my-runner ./scripts/nn.sh ...           # explicit override
NN_LOCAL=1 ./scripts/nn.sh ...                      # ignore run/remote.sh
```

Precedence:

1. `NN_RUNNER`, if set, receives every argument unchanged.
2. `run/remote.sh`, if that machine-local file exists and `NN_LOCAL` is not `1`, receives every
   argument unchanged, including `--cmd` and `--detach`.
3. Local `nn` (or `NN_BIN`).

The runner sources `run/env.sh` when it exists, and the local `--cmd` path runs `bash -c` with
the caller's environment, so `NN_STORE`, `NN_SERVER`, `NN_RUN`, `NN_LIBRARY`,
`CUDA_VISIBLE_DEVICES` and `HF_HOME` are inherited. The remote template forwards the same
variables. `run/` is gitignored; [../run.example/](../run.example/) holds the committed
templates and [remote.md](remote.md) explains the optional remote setup.

## Workflow config

The workflow is launched by the DSH `workflow` tool with an `args` object. The **Lead** — the
lead agent of an existing DeepSeek Harness (DSH) session — reads the meta JSON and the script
body and passes them as `meta` and `script`. Only changed keys are needed; these are the
defaults at the top of
[`.dsh/workflows/blind-neuron-test.js`](../.dsh/workflows/blind-neuron-test.js):

```js
{
  neurons: [],
  select: { n: 3, seed: 0, top: 20 },
  store: null,                  // required; no machine-specific default
  server: 'http://127.0.0.1:8765',
  runner: './scripts/nn.sh',
  run: 'main',
  work: null,                   // default <store parent>/blind-<run>
  reports: 'reports',
  batch: 3,
  inspectors: 1,
  testers: 3,
  paper: true,
  screen: null,
  steer: {},
  gpu: null,
}
```

| Key | Type | Meaning | Notes |
|---|---|---|---|
| `run` | string | label stamped on records | letters, digits, `_`, `.` or `-`; use a fresh one per experiment |
| `store` | path | harvested store | **required**; absolute path recommended, especially remotely |
| `server` | URL | `nn serve` address as seen by the runner | setup agent checks `/info` and the store's model |
| `runner` | command | prefix for every `nn` call | default `./scripts/nn.sh`; validated non-empty |
| `neurons` | string[] | explicit neuron ids | bypasses `select`; each is checked with `nn top` |
| `select` | object | `{ n, seed, top }` | random live neurons; `top: null` = all live |
| `screen` | object or null | intent screen and targeted harvest | harvests under the configured store's parent; the workflow uses the derived store |
| `work` | path or null | blind-test JSON directory | default `<store parent>/blind-<run>`; must differ from the store |
| `reports` | path | report folder in this checkout | default `reports`; gitignored |
| `batch` | integer | neurons in flight | ≥ 1; default 3 |
| `inspectors` | integer | independent inspectors per neuron | ≥ 1; a synthesizer runs only when > 1 |
| `testers` | integer | blind testers per pole | ≥ 1; default 3 |
| `paper` | boolean | write paper + library page | default true |
| `steer` | object | probe options | `{}` enables the default probe |
| `gpu` | number/null | `CUDA_VISIBLE_DEVICES` for screen/harvest/generate | server is not moved |

Validation covers `inspectors`, `testers`, `batch`, `screen.picks`, `run`, `runner` and every
`steer` field. `select` and the remaining `screen` fields are not fully validated: they are
passed to `nn`, where a bad value fails rather than before any agent runs. A missing `store`
throws before the screen or Setup.

### `select`

```js
select: { n: 10, seed: 1, top: 20 }
```

`nn select` samples `n` live (non-dead) neurons without replacement, optionally restricted to
the `top` live neurons by total fires (`top: null` uses all live neurons). The same seed gives
the same list. It prints ids; the workflow asks for JSON and returns the list unchanged.

### `screen`

```js
screen: {
  intent: 'violence',
  dataset: { id: 'YOUR/DATASET', config: null, split: 'train', field: 'text',
             where: ['label=harmful'], match: 'kill|attack' },
  benign:  { id: 'YOUR/DATASET', config: null, split: 'train', field: 'text',
             where: ['label=benign'] },
  picks: 3, top_k: 60, max_texts: 120, tokens: 100,
  harvest_tokens: 2e6,
  model: 'Qwen/Qwen3-0.6B',
  store: null,
}
```

| Key | Meaning | Default |
|---|---|---|
| `intent` | name copied into the screen output | empty |
| `dataset` | positive/intent source: `{ id, config, split, field, texts, where, match, max_docs }` | required |
| `benign` | benign source, same fields plus `match` | required |
| `picks` | neurons the agent carries forward | 3 |
| `top_k` | neurons written to the screen JSON | 60 |
| `max_texts` | texts per class | 120 |
| `tokens` | content tokens per text | 100 |
| `harvest_tokens` | tokens for the targeted harvest | 2e6 |
| `model` | model for screen and harvest | `Qwen/Qwen3-0.6B` |
| `store` | harvested store path | `<store parent>/store-<run>` |

`where` is a list of `KEY=VALUE` strings; matching is case-insensitive. `match` is a regex the
intent text must contain and `benign.match` (or `--benign-match`) is a regex the benign text
must contain: a text is kept only if it matches its source's regex, and both are optional.
`texts` can replace `id` for a local file. `max_docs` caps a source. The screen skips texts
shorter than `tokens`, needs at least two usable texts per class and at least three tokens per
text.

The screen agent writes the ranked JSON and picks file under `/tmp`, harvests the targeted
store, verifies each pick with `nn top`, and returns the store and list. The workflow derives
the targeted store path under the configured store's parent (default
`<store parent>/store-<run>`, or `screen.store`) and uses that derived store for the blind chain
and the probe; a store path returned by the agent is informational, not trusted.

### `steer`

```js
steer: {}   // the defaults below
```

| Key | Meaning | Default |
|---|---|---|
| `query` | ordinary user query used for the probe | `My roommate left dirty dishes in the sink again. What should I do?` |
| `ref` | dose unit for add/set: `sigma`, `absmax` or `resid` | `sigma` |
| `add` | strengths for `h += strength * unit` | `[8,16,32,64,128,256,-8,-16,-32,-64,-128,-256]` |
| `set` | strengths for `h = strength * unit` | same as `add` |
| `scale` | plain multipliers for `h *= scale` | `[1.5, 2, 3]` |
| `positions` | `all`, `new`, `prompt`; a string or a list | `['all', 'new']` |
| `max_new_tokens` | new tokens per generation | 70 |
| `temperature` | sampling temperature; 0 = greedy | 0.8 |

For `add` and `set`, a positive strength targets the `+` pole and a negative strength the `-`
pole. `scale` is a plain multiplier and preserves the sign of the activation already present,
so the positive-targets-plus rule applies to `add`/`set` only. Every strength must be finite;
duplicate strengths are removed. The workflow requires at least one of `add`, `set` or `scale`
to be non-empty. Unknown keys throw. The probe is one `nn generate --json --chat` call per
position policy carrying all doses, so calibration and BASE are shared. See
[workflow.md](workflow.md#stage-3--probe-and-save-a-showcase) for what the judge labels and how
the selection is made.

`steer` merges into the defaults key by key, so an unspecified `add`, `set` or `scale` keeps its
default sigma strengths. For residual-relative doses clear the lists you do not want:

```js
steer: { ref: 'resid', add: [], set: [0.3, 0.4, 0.5, 0.6], scale: [] }
```

With only `set` changed, the run would still apply the default 12 `add` values in resid units
and the 3 default `scale` multipliers, i.e. 19 doses instead of 4.

At the CLI, an empty mode list is legal: `nn generate ... --add '' --set '' --scale ''` returns
the calibration and BASE with no runs. That is useful for inspecting a prompt before choosing
doses; it cannot produce a showcase record on its own.

For late-layer neurons and short generations, the `all` and `new` position policies may be
indistinguishable in the generated text; that is a property of the probe, not a missing run.

## `nn` commands

### Find and harvest

`nn screen` — rank neurons by intent-vs-benign activation.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | model |
| `--intent` | empty | label copied into the JSON |
| `--out` | required | ranked JSON path |
| `--top-k` | 60 | neurons written |
| `--max-texts` | 120 | texts per class |
| `--tokens` | 100 | content tokens per text |
| `--batch` | 16 | texts per forward pass |
| `--device` | `cuda` | torch device |
| `--match` | none | regex the positive text must match |
| `--benign-match` | none | regex the benign text must match |
| `--min-mean` | 0.5 | drop neurons below this positive mean |
| `--dataset`, `--config`, `--split`, `--field`, `--where`, `--texts`, `--max-docs` | — | positive source |
| `--benign-dataset`, `--benign-config`, `--benign-split`, `--benign-field`, `--benign-where`, `--benign-texts`, `--benign-max-docs` | — | benign source |

`nn rank` — rank neurons by firing statistics and write `top.json`.

| Flag | Default | Meaning |
|---|---|---|
| `--out` | required | output directory (`fires.npy`, `sums.npy`, `peaks.npy`, `meta.json`, `top.json`) |
| `--model` | `Qwen/Qwen3-0.6B` | model |
| `--tokens` | 1e8 | tokens to stream |
| `--k` | 20 | neurons in `top.json` |
| `--stat` | `fires` | `fires`, `max`, `mean` or `spike` |
| `--fire` | 0.1 | `|h|` counted as a fire for `fires` |
| `--ctx` | 4096 | context sequence length |
| `--batch` | 8 | sequences per forward pass |
| `--checkpoint` | 5e7 | save interval |
| `--shard` | `0/1` | `i/n` share of the stream |
| `--device` | `cuda` | torch device |
| `--dataset…`, `--texts`, `--max-docs` | FineWeb | source |

`nn harvest` — stream a corpus into a store, resuming if it exists.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | model |
| `--tokens` | 1e8 | total tokens for this shard |
| `--ctx` | 4096 | context sequence length |
| `--batch` | 8 | sequences per forward pass |
| `--topk` | 20 | examples kept per neuron and regime |
| `--window` | 32 | tokens kept around each peak |
| `--before` | 24 | of which before the peak |
| `--fire` | 0.1 | `|h|` counted as a fire |
| `--checkpoint` | 5e7 | save interval |
| `--shard` | `0/1` | `i/n` share of the stream |
| `--device` | `cuda` | torch device |
| `--neurons` | none | JSON file: list or `{"neurons": [...]}`; harvest only those |
| `--dataset…`, `--texts`, `--max-docs` | FineWeb | source |

`--tokens` is a target, not a hard cap: sequence packing can overshoot it by up to
`batch × ctx`, and a finite corpus stops when it is exhausted. The same applies to `nn rank`.

`nn merge OUT INPUT...` combines shards harvested with the same settings ($k$, window, regimes,
model). `nn select N [--seed 0] [--top [20]] [--json]` samples live neurons.

### Inspect and record

| Command | What it does |
|---|---|
| `nn top NEURON [--pole +\|-] [--regime ++\|--\|+-\|-+\|mixed] [-n 10] [--start 0]` | strongest corpus examples, peak token in `[[ ]]` |
| `nn run NEURON --text TEXT [--min 0.1]` | per-token activations on any text |
| `nn steer NEURON --scale S --text TEXT [--top 8]` | next-token changes when the activation is multiplied |
| `nn ablate NEURON --text TEXT [--top 8]` | next-token and logit changes when the neuron is zeroed |
| `nn attrib NEURON --example <rank or text> [--pole +] [--top 15]` | upstream neurons and heads driving it |
| `nn generate NEURON --model MODEL --text TEXT [--chat] [--add LIST] [--set LIST] [--scale LIST] [--ref sigma\|absmax\|resid] [--positions all\|new\|prompt] [--max-new-tokens 80] [--temperature 0] [--seed 0] [--json]` | calibrated BASE plus one completion per dose; `--max-new-tokens` sets the length |
| `nn card submit NEURON --core TEXT [--negative TEXT] [--offshoots TEXT] [--residue TEXT]` | save a card; each field ≤ 500 characters; prints the id |
| `nn card redact CARD --pole +\|- --text TEXT` | store one pole's anonymized text (≤ 500 characters) |
| `nn card read CARD [--pole +\|-]` | print the card, or only the redacted text for a pole |
| `nn score FILE...` | score one test or a list of tests; writes per-test and pooled records |
| `nn paper NEURON` | print facts: card, redacted texts, newest scores, top examples, contrasts |
| `nn paper save NEURON --title ... --summary ... --clean 1-5 --clean-why ... --interesting 1-5 --interesting-why ... [--library ROOT]` | save a paper and update the library page |
| `nn steering save [--file FILE]` | validate one showcase record from stdin or a file |
| `nn report` | Markdown report of the newest scored neurons, papers and steering showcase |
| `nn library rebuild [NEURON ...] [--library ROOT]` | re-render pages from newest papers, scores and showcase records |
| `nn serve [--model MODEL] [--device cuda]` | load the model and serve the API at `NN_SERVER` |

`nn score` accepts one object or a list. Required: `neuron` or `card`, and `pole` (`+`/`-`).
Optional: `blind_max` (string or list), `new`, `near_miss`, `distant` (lists), `tester` (label),
`run` (default `--run`/`NN_RUN`). A text is strong at 30% of the pole's short-context maximum;
a verdict is WIN if new texts are strong more often than near misses with one-sided Fisher
exact p < 0.05, LOSS in the other direction, TIE otherwise. Several tests of one neuron, pole,
card and run in one call also write a pooled record.

### Corpus sources

`nn harvest`, `nn rank` and `nn screen` share the source flags:

| Flag | Meaning |
|---|---|
| `--dataset ID` | a Hugging Face dataset, streamed |
| `--config NAME` | dataset configuration (default: the dataset's) |
| `--split NAME` | split (default `train`) |
| `--field A,B` | string column(s) joined with newlines (default `text`) |
| `--where KEY=VALUE` | keep rows equal to the value, case-insensitive; repeatable |
| `--texts PATH` | a JSONL or plain-text file instead of a dataset |
| `--max-docs N` | stop after N documents (0 = all) |

A document is one dataset row or one file line. For a file, a JSON object uses its `text`
field(s); a bare JSON string or a plain line is the text. Blank lines are skipped. Rows with a
missing or non-string field raise an error rather than silently producing empty text. The
source spec is saved in `meta.json`, and a resume must use the same source and settings.

For a sharded harvest, each shard gets its own store and a different `--shard i/n`; merge with
`nn merge OUT SHARD...`. A targeted store (`--neurons`) must resume with the same list and cannot
be extended into a full store, or the reverse — use a new directory.

## Storage

A full store keeps per neuron, per regime, top-k peaks of a `window`-token context:

```
layers × neurons × 4 regimes × k × (4 + 4 × window) bytes
```

For `Qwen/Qwen3-0.6B` this is about 0.91 GB **even for a tiny corpus**, because the arrays are
dense over every layer and neuron. A targeted store for S neurons is
`S × 4 × k × (4 + 4 × window)`, but it still carries the tokenizer copy and a dense
`fires.npy`; for 20 neurons of that model the directory is about 13 MB. A 27B model
(64 layers × 17,408 neurons) at top-k 20 and window 32 needs about 11.8 GB, plus as much again
while a checkpoint is written. `vals.npy` is float32; `windows.npy` is int32; `fires.npy` is
int64 and stays dense even in a targeted store. `short_max.json` caches each pole's
short-context max, written by `nn score` on first use. Cards, scores, papers and steering
records are small JSON files.

Model VRAM is separate: weights plus activations, the harvest arrays and the generation cache.
`nn harvest` keeps `vals.npy`, `windows.npy` and `fires.npy` on the GPU while it runs, so the
store size above is also extra GPU memory on top of the weights: about 0.9 GB for Qwen3-0.6B
and about 11.8 GB for the 27B example. `nn generate` holds the model on one device and runs the
whole dose grid in a few calls; lower `max_new_tokens` or the dose grid if generation memory is
tight, and lower harvest `--batch`/`--ctx` if the harvest is tight. CPU is supported with
`--device cpu` but is much slower. Start with the small default model and one or two neurons.

## Paths and artifacts

| Path | Written by | Committed? |
|---|---|---|
| `$NN_STORE` | harvest, cards, scores, papers, steering; `short_max.json` (written by `nn score`) | no; gitignored via `runs/` and `data/` conventions |
| `<store parent>/blind-<run>/` | workflow Score/Setup work files | no |
| `reports/blind-<run>/` | workflow Report, manual report | no; `reports/` gitignored |
| `$NN_LIBRARY` (default `<store parent>/neuron-library/`) | `nn paper save`, `nn library rebuild` | no; `neuron-library/` gitignored |
| `run/` | your local runner config and logs | no; gitignored, templates are in `run.example/` |
