# Agent guide: running a blind neuron experiment

This is the working guide for an agent or operator running the three-stage pipeline.
[DESIGN_DOC.md](DESIGN_DOC.md) explains why it is built this way; [README.md](README.md) is the
user-facing overview; [docs/workflow.md](docs/workflow.md) has the launch protocol and the
manual equivalent.

## New session / new agent: start here

You need only this checkout and a shell. Run every command from the repository root. The
reference workflow needs an existing DeepSeek Harness (DSH) session; the manual path needs no
DSH.

1. **Install and check.**

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e .
   .venv/bin/nn --help
   ./scripts/nn.sh --check
   ```

   `./scripts/nn.sh --check` works locally and prints the `nn` binary, Python version, store
   status, GPU information and free disk. The first install downloads dependencies, so it needs
   network access. The model loads in bf16; CPU (`--device cpu`) is slow but works for smoke
   tests.

2. **Pick a dataset and build or choose a store.** A local file is the fastest start:

   ```bash
   export NN_STORE="$PWD/runs/my-store"
   .venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e6 \
     --texts /path/to/corpus.jsonl --max-docs 200
   ```

   A Hugging Face dataset works the same way with `--dataset ID [--config] [--split] [--field]
   [--where KEY=VALUE]`. To keep the store small, screen first and harvest only the picks:

   ```bash
   .venv/bin/nn screen --model Qwen/Qwen3-0.6B --intent YOUR_INTENT \
     --texts intent.jsonl --benign-texts benign.jsonl \
     --out /tmp/screen.json --top-k 60 --max-texts 120 --tokens 100
   # read /tmp/screen.json, write /tmp/picks.json as a list or {"neurons": [...]}
   .venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e6 \
     --texts /path/to/corpus.jsonl --neurons /tmp/picks.json
   ```

   A full Qwen3-0.6B store is about 0.9 GB even for a tiny corpus; a targeted store is about
   13 MB. If you already have a store, skip the harvest and draw neurons with
   `.venv/bin/nn select 3 --seed 0 --top 20 --json`.

3. **Start the server.** The blind chain needs `nn serve`; the workflow does not start or stop
   it. Start it yourself and wait until `/info` answers:

   ```bash
   CUDA_VISIBLE_DEVICES=YOUR_GPU .venv/bin/nn serve
   # another shell:
   curl -fsS "$NN_SERVER/info"
   ```

4. **Run the workflow (reference) or the manual path.** The workflow is not a command name. The
   **Lead** is the lead agent of an existing DSH session. The Lead reads
   `.dsh/workflows/blind-neuron-test.meta.json` and `.dsh/workflows/blind-neuron-test.js` and
   calls the `workflow` tool with `meta` set to the meta JSON object, `script` set to the `.js`
   body, and `args` set to your run config. `store` is required and has no machine-specific
   default; `runner` defaults to `./scripts/nn.sh`. A minimal config:

   ```js
   {
     run: 'r1',
     store: '/absolute/path/to/store',
     select: { n: 3, seed: 0, top: 20 },
     runner: './scripts/nn.sh',
     gpu: 0,
   }
   ```

   Without DSH, [docs/workflow.md](docs/workflow.md#manual-equivalent) is the supported and
   complete manual equivalent: `nn select`, `nn card submit/redact/read`, blind testers,
   `nn score`, `nn paper save`, `nn generate`, `nn steering save`, `nn library rebuild` and
   `nn report`.

5. **Find the outputs.** The store keeps `cards/`, `scores/`, `papers/` and `steering/`; the
   blind-test JSON goes to `<store parent>/blind-<run>/`; the library goes to
   `<store parent>/neuron-library/` (or `NN_LIBRARY`); the report and the run's copied pages go
   to `reports/blind-<run>/` in the calling checkout. `reports/`, `runs/` and `run/` are
   gitignored.

## Install

Python 3.10+ (3.10, 3.11 and 3.12 are tested) and the dependencies in `pyproject.toml` (torch,
transformers, datasets, accelerate, numpy, tokenizers). The model must use a gated MLP with
`gate_proj`, `up_proj` and `down_proj`; the code reads `act_fn(gate_proj(x)) * up_proj(x)`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/nn --help
./scripts/nn.sh --check
```

The distribution name is `inspect-neuron`; the console command is `nn`. The first install needs
network access. pip may install a CPU-only torch wheel; for CUDA, install the matching wheel from
<https://pytorch.org/get-started/locally/> first.

## Runner

`./scripts/nn.sh` is the one runner every workflow command uses. It runs `nn` locally by
default (preferring `.venv/bin/nn`) and delegates every argument, `--cmd` included, to
`run/remote.sh` if that machine-local file exists. The template is in `run.example/`; `run/` is
gitignored.

```bash
./scripts/nn.sh --help
./scripts/nn.sh --check                  # local: nn, python, store, GPU, disk
./scripts/nn.sh --cmd 'cat "$NN_STORE/meta.json"'
./scripts/nn.sh --dry-run top L14.N957 --pole +
./scripts/nn.sh --detach serve           # local or remote: prints pid and log path
NN_RUNNER=./my-runner ./scripts/nn.sh top L14.N957 --pole +
NN_LOCAL=1 ./scripts/nn.sh top L14.N957 --pole +    # ignore run/remote.sh
NN_BIN=/path/to/nn ./scripts/nn.sh --check          # choose the executable
```

Precedence: `NN_RUNNER` if set, else `run/remote.sh` if it exists and `NN_LOCAL` is not `1`,
else local `nn`. Nothing in this repository names a GPU host. Local mode is the default; see
[docs/remote.md](docs/remote.md) for the optional remote setup. If you need GPU pinning, use
`CUDA_VISIBLE_DEVICES` or the workflow's `gpu` option.

## Environment

Every `nn` command reads these variables; flags override them.

| Variable | Meaning | Default |
|---|---|---|
| `NN_STORE` | store directory (harvest output, cards, scores, papers, steering records) | none, required |
| `NN_SERVER` | URL of the running `nn serve` | `http://127.0.0.1:8765` |
| `NN_RUN` | run label stamped on cards, scores, papers and steering records | none |
| `NN_LIBRARY` | root of the neuron library (HTML pages) | `neuron-library/` beside the store |
| `HF_HOME` | Hugging Face model and dataset cache | the library default |
| `CUDA_VISIBLE_DEVICES` | GPU(s) seen by `harvest`, `screen`, `rank`, `serve` and `generate` | all visible devices |

`scripts/nn.sh` forwards `NN_STORE`, `NN_SERVER`, `NN_RUN`, `NN_LIBRARY`,
`CUDA_VISIBLE_DEVICES` and `HF_HOME` for every call, including `--cmd`. Global flags go before
the subcommand (`nn --store PATH --run r1 report`); `nn --help` and `nn <command> --help` list
every flag.

## Machine rules

- GPUs can be shared. Never kill, restart or attach to a process you did not start.
- Check `nvidia-smi` before a long job, pick a free device, and delete only the stores, shards
  and runs you created.
- `nn harvest` streams the corpus. Do not download or save a web corpus to disk; point
  `HF_HOME` at a disk with room and keep stores off a full volume.
- Harvest keeps the store arrays on the GPU. Budget the store size as GPU memory too: about
  0.9 GB extra for Qwen3-0.6B and about 11.8 GB extra for a 27B model, on top of the weights.
- Keep paths absolute when the runner may target another machine.

## Harvest and rank

A full harvest (FineWeb by default):

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU NN_STORE=/path/to/store ./scripts/nn.sh harvest \
  --model Qwen/Qwen3-0.6B --tokens 2e7
```

Defaults: `--topk 20` per regime, `--window 32` (`--before 24`), `--ctx 4096`, `--batch 8`, a
checkpoint every 5e7 tokens. Rerunning with a larger `--tokens` resumes. `--tokens` is a target,
not a hard cap: packing can overshoot by up to `batch × ctx`, and a finite corpus stops when it
is exhausted. A full Qwen3-0.6B store is about 0.9 GB; a targeted store is about 13 MB. Store
size arithmetic is in [DESIGN_DOC.md](DESIGN_DOC.md#storage).

Any corpus can replace FineWeb: `--dataset <hf id> [--config] [--split] [--field]
[--where KEY=VALUE]`, or `--texts <jsonl-or-txt>`; `--max-docs` caps a run. A local `.jsonl`
or `.txt` file has one document per line.

To keep only the neurons that fire most, rank first and harvest a targeted store. `nn rank`
streams the dataset once without a store and writes `fires.npy`, `sums.npy`, `peaks.npy`,
`meta.json` and `top.json`
(the top `--k` neurons, default 20); `nn harvest --neurons` keeps only those:

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU NN_STORE=/path/to/store ./scripts/nn.sh rank \
  --out /path/to/rank --model Qwen/Qwen3-0.6B --tokens 2e7 --k 20
CUDA_VISIBLE_DEVICES=YOUR_GPU NN_STORE=/path/to/store ./scripts/nn.sh harvest \
  --model Qwen/Qwen3-0.6B --tokens 2e7 --neurons /path/to/rank/top.json
```

`--stat` on `nn rank` picks the ranking: `fires` (default; tokens above `--fire`), `max` (peak
magnitude), `mean` (mean positive activation) or `spike` (max - mean). On a corpus where most
neurons fire on almost every token, `fires` and `mean` select the always-on neurons; `spike`
selects the neurons the corpus actually triggers.

`nn screen` is the intent-specific front end: it ranks neurons by intent-vs-benign activation
and writes JSON with a Welch t and peak tokens. `nn harvest --neurons` consumes its picks.
Texts shorter than `--tokens` are skipped, and each class needs at least two usable texts.
`nn top` values are corpus peaks from long documents and can be much larger than a short
standalone text's activation; the blind test scores short texts against a short-context max, so
check candidates with `nn run` before choosing examples.

For a large model, run one shard per GPU into separate stores, then merge them (all shards need
the same settings):

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU NN_STORE=/path/to/store/shard0 ./scripts/nn.sh harvest \
  --model Qwen/Qwen3-0.6B --tokens 1e7 --shard 0/2
# a second GPU does the other half into /path/to/store/shard1 with --shard 1/2
./scripts/nn.sh merge /path/to/store /path/to/store/shard0 /path/to/store/shard1 ...
```

## Server

The workflow does not start or stop the server. Start it yourself and wait until it answers:

```bash
CUDA_VISIBLE_DEVICES=YOUR_GPU NN_STORE=/path/to/store ./scripts/nn.sh serve
# another shell:
curl -fsS "$NN_SERVER/info"
```

The model loads from the store's `meta.json` unless `--model` is given. Wait until `/info`
answers before starting a run. Stop only the server you started. The blind chain needs the
server; `nn screen`, `nn harvest`, `nn generate`, `nn report` and `nn library rebuild` do not.

## Run the workflow

The workflow is two files: `.dsh/workflows/blind-neuron-test.js` (the script body) and
`.dsh/workflows/blind-neuron-test.meta.json` (the `workflow` tool's meta). DSH has no
name-based launcher: the Lead (the lead agent of an existing DSH session) reads both files and
calls the `workflow` tool with `meta` set to the meta JSON object, `script` set to the `.js`
body, and `args` set to the run config. See [docs/workflow.md](docs/workflow.md) for the exact
protocol and the manual equivalent.

A full intent-first run (screen, blind test, probe, report):

```js
{
  run: 'r3',
  store: '/path/to/base-store',
  screen: {
    intent: 'violence',
    dataset: { id: 'PKU-Alignment/BeaverTails', split: '330k_train', field: 'response',
               where: ['is_safe=false'], match: 'kill|stab|shoot|bomb|attack|weapon|gun' },
    benign:  { id: 'PKU-Alignment/BeaverTails', split: '330k_train', field: 'response',
               where: ['is_safe=true'] },
    picks: 3, top_k: 60, max_texts: 120, tokens: 100, harvest_tokens: 2e6,
  },
  steer: {
    query: 'My roommate left dirty dishes in the sink again. What should I do?',
    ref: 'sigma',
    add: [8, 16, 32, 64, 128, 256, -8, -16, -32, -64, -128, -256],
    set: [8, 16, 32, 64, 128, 256, -8, -16, -32, -64, -128, -256],
    scale: [1.5, 2, 3],
    positions: ['all', 'new'], max_new_tokens: 70, temperature: 0.8,
  },
  gpu: 0,
}
```

Without `screen`, pick from an existing store by name or at random:

```js
{ run: 'r1', store: '/path/to/store', select: { n: 10, seed: 1 } }
{ run: 'r2', store: '/path/to/store', neurons: ['L14.N957'], inspectors: 3 }
```

Neuron ids in this guide are illustrative; use ids from your own store.

`steer: {}` is the default probe: sigma units, add/set ±8..256, scale [1.5, 2, 3], positions
all and new, 70 new tokens at temperature 0.8 and seed 0. `steer` merges into those defaults
key by key, so an `add`, `set` or `scale` list you do not pass keeps its default sigma
strengths. For residual-relative doses clear the lists you do not want and pass the small resid
values, for example
`steer: { ref: 'resid', add: [], set: [0.3, 0.4, 0.5, 0.6], scale: [] }`; leaving the default
`add` (12 values) and `scale` (3 values) in place would run 15 extra doses, 12 of them
meaninglessly large adds in resid units, alongside the 4 intended ones.

`store` is required and has no machine-specific default; it must exist unless `screen` harvests
a targeted one. The workflow's `runner` option (default `./scripts/nn.sh`) prefixes every `nn`
command. The workflow does not read `NN_STORE` from the shell for its own config.

The workflow's meta declares 10 stage names: Screen, Setup, Inspect, Synthesize, Anonymize,
Test, Score, Paper, Steer and Report. Only Screen, Setup, Steer and Report call `phase()`, so
the progress display has four phases and the other stages run inside them. Each agent task
carries a run-prefixed `opts()` label naming the task, such as `r1:inspect:L14.N957`, not the
stage title, and the `nn steering save` recorder adds a `showcase` label, so there are 11 label
kinds for the 10 stages. Synthesize is skipped when `inspectors` is 1.

## What each stage does

1. **Find.** `screen` is optional: `nn screen` ranks all neurons by intent-vs-benign
   activation, an agent picks `picks` of them from the ranked JSON and their peak tokens, and
   `nn harvest --neurons` builds a targeted store under the configured store's parent (default
   `<store parent>/store-<run>`); the workflow uses that derived store. Without `screen`,
   `neurons` names them (checked with `nn top`) or `select: { n, seed, top }` uses `nn select`
   to draw from the K hottest live neurons by total fires (`top` defaults to 20; `top: null`
   uses all live neurons).
2. **Blind test.** Setup checks the store and server and resolves the list. For each neuron,
   `inspectors` agents explain both poles independently; with more than one, a synthesizer
   merges them into one card. One anonymizer per pole removes every leak, `testers` blind
   testers per pole write texts from the anonymized description alone with no tools, and one
   pooled `nn score` call judges all tests. An optional paper writer adds the paper and library
   page.
3. **Probe.** Every neuron with a successful score gets a fixed calibrated probe: one
   `nn generate --json --chat` call per position policy applies the whole add/set/scale grid,
   so one calibration and one BASE are shared. `ref` is the dose unit: `sigma` is the prompt's
   activation standard deviation, `absmax` its absolute activation maximum, `resid` the layer's
   mean residual-stream norm divided by the neuron's write-column norm. For `add` and `set`,
   positive strengths target the + pole and negative strengths the - pole; `scale` is a plain
   multiplier that preserves the sign of the existing activation. One judge agent per neuron
   reads the compact digest of the explicit JSON files (never a glob) and labels every run
   `coherent`, `on_concept`, `unaligned` and `movement` 0-3; it can fetch a 300-character-cut
   run in full before judging. The workflow selects deterministically: only coherent on-concept
   runs are candidates, first by largest `|strength|`, then `movement`, then position all > new >
   prompt, then mode set > add > scale. An incoherent run can never be selected; when there is
   none it says so and names the strongest coherent run. It then saves one showcase record per
   neuron with `nn steering save` when a judged candidate and a calibration exist: the selected
   run, a coherent control when one exists (otherwise `null`), the full generation, the judge's
   classification, the recipe and the reproduce command. Coherent unaligned runs are recorded
   as evidence. If no candidate exists, no record is saved and the `## Steering` section says
   so.
4. **Report.** `nn report` renders the blind-test body and, when a steering record was saved,
   `## Showcase`; the Report agent
   rebuilds the library pages so they carry the newest showcase record, copies the pages whose
   embedded run matches this run into `reports/blind-<run>/` using the paths `nn library rebuild`
   prints, then appends `## Steering` (one calibration line per position, the selection,
   coherent unaligned evidence and a `Pos | Mode | Strength | ...` table).

## Configuration

Only changed keys are needed; the defaults are at the top of the workflow and in
[DESIGN_DOC.md](DESIGN_DOC.md#configuration).

- `run` is the label stamped on cards, scores, papers and steering records; use a fresh one per
  run.
- `store` is the harvested store, wherever the runner points; `reports` is the folder in this
  checkout for `report.md` and the HTML pages.
- `runner` prefixes every nn command (default `./scripts/nn.sh`: local unless `run/remote.sh`
  exists).
- `neurons` lists names; `select: { n, seed, top }` picks at random; `screen: null` (default)
  skips the screen.
- `batch` is the number of neurons in flight (default 3).
- `inspectors` is the inspector count per neuron (default 1; a synthesizer runs only above 1).
  `testers` is the blind tester count per pole (default 3). `paper: false` skips papers.
- `steer` is the probe (default `{}`): `query`, `ref` (`sigma`/`absmax`/`resid`), `add`, `set`,
  `scale`, `positions` (`all`/`new`/`prompt`), `max_new_tokens`, `temperature`. Unknown keys
  throw, so a stale option fails loudly instead of being ignored.

Validation covers `inspectors`, `testers`, `batch`, `screen.picks`, `run`, `runner` and every
`steer` field. `select` and the remaining `screen` fields are not fully validated: they are
passed to `nn`, where a bad value fails. `store` is required.

The `screen` round derives the targeted store under the configured store's parent (default
`<store parent>/store-<run>`, or `screen.store`); the workflow uses that derived store, not a
path returned by the screen agent. Texts shorter than `tokens` are skipped.

At the CLI, empty mode grids are legal and produce calibration plus BASE only:
`nn generate ... --add '' --set '' --scale ''`. The workflow's `steer` config still needs at
least one dose.

Agents run on the harness's configured default model; the workflow has no per-role model or
effort setting. Agents must not post to chat or send messages; their final output is their only
report. They must not write files into this checkout either: scratch goes under `/tmp`, and
only the Report phase writes under `reports/blind-<run>/`.

## Commands

- Inspect: `nn top`, `nn run`, `nn steer`, `nn ablate`, `nn attrib`, `nn generate`.
- Record: `nn card submit`, `nn card redact`, `nn card read`, `nn score`, `nn paper`,
  `nn paper save`, `nn steering save`.
- Find: `nn screen`, `nn select`, `nn rank`, `nn harvest`, `nn merge`.
- Report: `nn report`, `nn library rebuild`.
- Serve: `nn serve`.
- `nn --help` and `nn <command> --help` list every flag.

## Outputs

The store, work directory and neuron library live where the runner points (this machine by
default; the GPU host in remote mode). The report and the run's copied pages are written in the
calling checkout, because their redirection is local. Pull the remote library or logs with
`rsync` if you want the whole index ([docs/remote.md](docs/remote.md)).

- `<store>/cards/`, `<store>/scores/`, `<store>/papers/`, `<store>/steering/`: JSON records
  stamped with the run label. A steering record is one showcase generation (`nn steering
  save`); the report and library read the newest one per neuron, and per run with `--run`.
  `run` and `model` are optional in a record: `run` defaults to `--run`/`NN_RUN`, and `model`
  defaults to the store's model. `control` may be `null`.
- `<store>/short_max.json`: cache of each pole's short-context max, written by `nn score` and
  reused by later scores.
- `<work>/` (default `<store parent>/blind-<run>`): the blind-test JSON files.
- `reports/blind-<run>/`: `report.md` (body + `## Showcase` when a record exists + `## Steering`
  in the DSH workflow) and each winning
  neuron's `<neuron>.html` page. `reports/` is gitignored.
- `$NN_LIBRARY/<model>/<neuron>.html` (default `<store parent>/neuron-library/`): one page per
  neuron with a winning pole, plus `index.html`. `NN_LIBRARY` or `--library` sets another root.
  `nn paper save` writes or removes the page; a run-scoped `nn library rebuild` removes a page
  only if that run owns it or no other run still has a winning score for the neuron, so
  rebuilding an old run never deletes a newer run's page. `nn library rebuild` re-renders from
  the newest papers, scores and steering records and needs no server.

## Limits

- There is no ground truth for what a neuron means. A WIN is predictive evidence, not proof.
- The judge reads a 300-character digest of each generation and can fetch a cut run in full; it
  labels movement coarsely and is a filter, not a measurement.
- One query can be unrepresentative; pass a query that matters to the concept.
- Working dose windows are narrow and non-monotonic. "No coherent on-concept dose" is a real
  result, recorded as such, not a missing run.
- `Qwen/Qwen3-0.6B` is a smoke-test model; results need not transfer to larger models.
- The blind chain needs a running `nn serve`; the probe, report and library do not.
- Anonymization and judging are LLM judgments; reruns are not deterministic.
