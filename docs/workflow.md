# Running the workflow

The three stages — find neurons, blind-test explanations, probe steering — can be run in two
ways:

1. **DSH workflow (reference).** The two files in `.dsh/workflows/` are read by the **Lead**,
   the lead agent of an existing DeepSeek Harness (DSH) session, which calls the `workflow`
   tool. This runs the agent chain: inspectors, anonymizers, blind testers, score writer, paper
   writer, steering judge and report writer. The workflow needs an existing DSH session; it
   cannot start one. DSH is at <https://github.com/deepseek-ai/deepseek-harness>.
2. **Manual equivalent.** Run the same `nn` commands yourself, with a person or a separate
   model playing each blind role. This is more work but needs no DSH and is the supported route
   when you do not have a DSH session.

Both paths use the same store, records, report and library. The workflow's meta declares
**10 stage names**, but only 4 of them are progress phases (Screen, Setup, Steer, Report); its
agents carry run-prefixed task labels (`r1:inspect:L14.N957`), not the stage titles. Start with
[../README.md](../README.md) for installation and the quickstart.

## Before you start

- A harvested store (see the quickstart). The workflow never harvests unless you give it a
  `screen` config. `store` is required and has no machine-specific default; pass an absolute
  path.
- A running `nn serve` for the blind chain. Start it yourself; the workflow only checks it.
- The store and server must agree on the model (`<store>/meta.json` and `/info`).
- Use an **absolute** store path when the workflow runs through a remote runner.
- A fresh `run` label for every experiment. Records are stamped with it, and `--run` limits
  lookups to that label.

The workflow's `runner` option (default `./scripts/nn.sh`) prefixes every `nn` command.
Locally that is `nn`; if `run/remote.sh` exists it is your GPU host. See
[remote.md](remote.md). Global flags (`--store`, `--server`, `--run`) go **before** the
subcommand: `nn --store /path/to/store --run r1 report`.

## Launch through DSH

DSH has no name-based launcher. The Lead (the lead agent of an existing DSH session) reads both
files and calls the `workflow` tool with:

- `meta`: the JSON object from
  [`.dsh/workflows/blind-neuron-test.meta.json`](../.dsh/workflows/blind-neuron-test.meta.json);
- `script`: the contents of
  [`.dsh/workflows/blind-neuron-test.js`](../.dsh/workflows/blind-neuron-test.js) (the body,
  without an `export const meta` statement);
- `args`: your run configuration object.

In a DSH session, the request can be as short as:

> Run the blind-neuron-test workflow with args
> `{ run: 'r1', store: '/path/to/store', select: { n: 3, seed: 0 } }`.

The Lead is expected to read both files from disk for each run; do not paraphrase the script
body or edit the meta in the request. The meta JSON and the script must stay in sync: the meta
declares ten stage titles, `phase()` is called only for Screen, Setup, Steer and Report, and the
test suite checks that each agent's `opts()` phase is one of those titles, in meta order. The
labels themselves are run-prefixed and task-specific (`r1:inspect:L14.N957`); the
`nn steering save` recorder adds a `showcase` label in the Steer phase, so there are 11 label
kinds for the 10 stages.

The script body is about 51 KB. Pass it byte-for-byte: a truncated or mistranscribed copy usually
fails to parse (`SyntaxError`), which is a delivery error rather than a workflow result — but a
body that parses is not proof that it matches the file byte-for-byte. If the harness cannot carry
a string that large reliably, use the [manual equivalent](#manual-equivalent) below, which runs
the same stages.

### Minimal args (existing store)

```js
{
  run: 'r1',
  store: '/path/to/store',
  select: { n: 3, seed: 0, top: 20 },
  runner: './scripts/nn.sh',
  gpu: 0,
}
```

`select` draws from the live neurons with the most total fires. `top: null` draws from all live
neurons. A rerun with the same seed gives the same list. To name neurons yourself:

```js
{ run: 'r2', store: '/path/to/store', neurons: ['L14.N957', 'L3.N42'], inspectors: 3 }
```

Explicit ids are checked with `nn top`; a missing neuron stops Setup with `ok: false`.

### Screen-first args (harvest a targeted store)

```js
{
  run: 'r3',
  store: '/path/to/base-store',       // parent for the derived targeted store: <parent>/store-r3
  screen: {
    intent: 'violence',
    dataset: { id: 'YOUR/DATASET', split: 'train', field: 'text',
               where: ['label=harmful'], match: 'kill|attack|weapon' },
    benign:  { id: 'YOUR/DATASET', split: 'train', field: 'text',
               where: ['label=benign'] },
    picks: 3, top_k: 60, max_texts: 120, tokens: 100, harvest_tokens: 2e6,
  },
  steer: {
    query: 'My roommate left dirty dishes in the sink again. What should I do?',
    ref: 'sigma',
    add: [8, 16, 32, 64, 128, 256, -8, -16, -32, -64, -128, -256],
    set: [8, 16, 32, 64, 128, 256, -8, -16, -32, -64, -128, -256],
    scale: [1.5, 2, 3],
    positions: ['all', 'new'],
    max_new_tokens: 70,
    temperature: 0.8,
  },
  gpu: 0,
}
```

The screen agent runs `nn screen`, reads the ranked JSON and the peak tokens, picks `picks`
neurons, writes a picks file under `/tmp`, harvests the targeted store and verifies each pick
with `nn top`. The workflow derives the targeted store path under the configured store's parent
(default `<store parent>/store-r3`) and uses that derived store for the blind chain and the
probe; the `store` field the agent returns is informational, not trusted. With `screen`,
`store` is required but used as the parent anchor; the base store itself need not exist.
`screen.store` can override the derived path. Texts shorter than `tokens` are skipped. If fewer
usable neurons are found, the screen stops before Setup with a clear problem.

The default probe is used when `steer` is `{}` or omitted: sigma units, add/set ±8..256,
scale 1.5/2/3, positions `all` and `new`, 70 new tokens at temperature 0.8. `steer` merges into
those defaults key by key, so any list you leave out keeps its default sigma strengths. For
residual-relative doses clear the lists you do not want and pass the small resid values, for
example `steer: { ref: 'resid', add: [], set: [0.3, 0.4, 0.5, 0.6], scale: [] }`; with the
default `add` and `scale` still in place the run does 15 extra doses, 12 of them meaninglessly
large adds in resid units, alongside the 4 intended ones. Unknown `steer` keys throw before any
agent runs.

### What the stages do

The meta declares 10 stage names but only 4 progress phases (Screen, Setup, Steer, Report); the
other stages run inside one of those phases and their agents carry run-prefixed task labels
(`r1:score:L14.N957`) rather than the stage title. The 11th label kind is `showcase`, the
`nn steering save` recorder in the Steer phase.

| Stage | What happens |
|---|---|
| Screen | optional: rank neurons for the intent against a benign contrast, pick, harvest a targeted store |
| Setup | check the server and `<store>/meta.json`, resolve the neuron list, make the work directory |
| Inspect | `inspectors` agents explain both poles independently from the corpus and the `nn` tools |
| Synthesize | only when `inspectors > 1`: merge the findings into one card |
| Anonymize | one agent per pole removes examples, numbers, ids, method names and pole sign, then `nn card redact` |
| Test | `testers` blind agents per pole see only the redacted text and have no tools |
| Score | one pooled `nn score` call per neuron |
| Paper | optional: `nn paper` facts, then `nn paper save` (also writes the library page) |
| Steer | one calibrated `nn generate --json --chat` call per position policy, one judge per neuron, deterministic selection, one `nn steering save` record per neuron when a candidate and calibration exist |
| Report | `nn report`, `nn library rebuild`, copy the run's pages, append `## Steering` |

The Setup, Screen and Report agents run ordinary shell commands through the same runner, so
`--cmd` must work. `./scripts/nn.sh --cmd '...'` handles that locally; the remote template
forwards it.

### Failure and resume behaviour

- A failed Screen or Setup returns `{ ok: false, problem, ... }` and no neuron is tested.
- A failed neuron is listed in `failed`; other neurons continue.
- A missing or unparseable test is skipped, and the workflow logs how many blind tests were
  dropped.
- Re-running with the same `run` label can add newer records; reports and pages use the newest
  record per neuron (per run when `--run` is set). Use a fresh label when you want a clean
  comparison.
- The workflow does not start, stop or restart the server, and agents are told not to touch
  processes they did not start.

### A note on agents

The workflow agents run on whatever model and provider the DSH session is configured for; the
workflow itself has no per-role model setting. Agents are instructed not to post to chat and
not to write files in the checkout, except the Report phase writing under
`reports/blind-<run>/`. If you change those harness defaults, the scientific protocol is
unchanged, but your texts and judgments will differ.

## Manual equivalent

This path runs the same commands without DSH. The important discipline is **blindness**:
whoever writes the tests must see only the anonymized text.

When the runner is remote, every path you pass to `nn` must exist on the machine where `nn`
runs, not just in this checkout: a `/tmp/picks.json` or `/tmp/L14.N957-tests.json` written here
is not visible there. Write it through the runner, whose stdin is kept:

```bash
./scripts/nn.sh --cmd 'cat > /tmp/picks.json' <<'EOF'
{"neurons": ["L14.N957", "L3.N42"]}
EOF
```

### Stage 1 — find neurons

Neuron ids in this page (for example `L14.N957` and `L3.N42`) are illustrative; use ids from
your own store.

If you already have a store, list or sample neurons:

```bash
.venv/bin/nn select 3 --seed 0 --top 20 --json
# or: nn select 3 --seed 0 --top 20
```

To screen, either point at local files (one document per line) or at a Hugging Face dataset:

```bash
.venv/bin/nn screen --model Qwen/Qwen3-0.6B --intent YOUR_INTENT \
  --texts intent.jsonl --benign-texts benign.jsonl \
  --out /tmp/screen.json --top-k 60 --max-texts 120 --tokens 100

.venv/bin/nn screen --model Qwen/Qwen3-0.6B --intent YOUR_INTENT \
  --dataset YOUR/DATASET --split train --field text --where label=harmful \
  --benign-dataset YOUR/DATASET --benign-split train --benign-field text --benign-where label=benign \
  --out /tmp/screen.json --top-k 60 --max-texts 120 --tokens 100
```

Texts shorter than `--tokens` are skipped, and the screen needs at least two usable texts per
class. The JSON contains, per neuron, `neuron`, `t`, `mean_pos`, `max_pos`, `mean_neg`, `max_neg`,
`peak_pos` and `peak_neg`. Pick candidates whose `t` is high, whose positive mean is well above
the benign mean, whose positive maximum is at least about 5, and whose peak tokens are content
words rather than punctuation, digits or bare function words. Write the picks as either a list
or `{"neurons": [...]}`:

```bash
cat > /tmp/picks.json <<'EOF'
{"neurons": ["L14.N957", "L3.N42"]}
EOF
.venv/bin/nn harvest --model Qwen/Qwen3-0.6B --tokens 2e6 \
  --dataset YOUR/DATASET --split train --field text --neurons /tmp/picks.json
```

To harvest the full model instead, omit `--neurons`. `nn rank --out /path/to/rank ...` writes a
`top.json` that `--neurons` also accepts. `nn harvest` resumes if the store exists, and stops at
the end of a local corpus; `--tokens` is a target and can overshoot by up to `batch × ctx`.

`nn top` values are corpus peaks measured inside long documents and can be much larger than a
short standalone text's activation. The blind test scores short texts against a short-context
max, so verify a candidate with `nn run` on a short text before choosing examples.

### Stage 2 — explain and blind-test

The ids below are illustrative. For each neuron, inspect it yourself (or with an agent) using
the full data:

```bash
.venv/bin/nn top L14.N957 --pole + -n 20
.venv/bin/nn top L14.N957 --pole - --regime mixed -n 20
.venv/bin/nn run L14.N957 --text "a text you wrote to test the hypothesis"
.venv/bin/nn steer L14.N957 --scale 2 --text "..."
.venv/bin/nn ablate L14.N957 --text "..."
.venv/bin/nn attrib L14.N957 --pole + --example 1
```

Describe each pole as a verb: what it does, which texts and tokens fire it, and what does not.
Write the card:

```bash
.venv/bin/nn card submit L14.N957 \
  --core "Explains the + pole..." \
  --negative "Explains the - pole, or empty..." \
  --offshoots "Known related behaviour..." \
  --residue "What remains unexplained..."
```

Now anonymize each pole. Use a **different agent or person** from the one who wrote the card.
The redacted text must not contain examples from the card, numbers or measurements, neuron,
layer or head ids, method names (steering, ablation, attribution, corpus max) or the pole sign:

```bash
.venv/bin/nn card redact <CARD_ID> --pole + --text "An abstract description of the concept..."
.venv/bin/nn card read <CARD_ID> --pole +       # this is exactly what the tester may see
```

Give only that output to the blind tester, with no tools and no other context. The tester
writes one `blind_max`, five `new` examples, five `near_miss` twins (same order, minimal edits
that change only what the explanation says matters) and five `distant` texts for the most
distant concept the description still covers. Put the result in a JSON file:

```json
[
  {
    "neuron": "L14.N957",
    "pole": "+",
    "tester": "t1",
    "run": "r1",
    "blind_max": "One text that should fire it hardest.",
    "new": ["...", "...", "...", "...", "..."],
    "near_miss": ["...", "...", "...", "...", "..."],
    "distant": ["...", "...", "...", "...", "..."]
  }
]
```

`card` may replace `neuron`; `run` defaults to `--run`/`NN_RUN`; `tester` is a label. A single
object is accepted as well as a list. Score each tester separately, but in one call per neuron
so the pole is pooled:

```bash
.venv/bin/nn score /tmp/L14.N957-tests.json
```

`nn score` stores one record per test (and, when several tests share a neuron, pole, card and
run, a pooled record), judges a text strong when it reaches 30% of the pole's short-context
maximum, and reports WIN/TIE/LOSS from one-sided
Fisher exact tests on new versus near-miss counts. Read the newest score for a neuron with:

```bash
.venv/bin/nn paper L14.N957
```

Then write the paper (not blind — you may read everything):

```bash
.venv/bin/nn paper save L14.N957 \
  --title "A short title naming what the neuron does" \
  --summary "One paragraph: what it does and how well the blind test supports it." \
  --clean 4 --clean-why "How fully one simple concept explains it." \
  --interesting 3 --interesting-why "Why this neuron is interesting."
```

Repeat for the other pole with a separate redaction and separate tests. A pole that never
fires can be left empty in the card and is not tested.

### Stage 3 — probe and save a showcase

Run one calibrated sweep. In the reference workflow, one call per position policy carries the
whole grid, so calibration and BASE are shared:

```bash
.venv/bin/nn generate L14.N957 --model Qwen/Qwen3-0.6B --chat \
  --text "Your probe query" \
  --add 8,16,32,64,128,256,-8,-16,-32,-64,-128,-256 \
  --set 8,16,32,64,128,256,-8,-16,-32,-64,-128,-256 \
  --scale 1.5,2,3 \
  --ref sigma --positions all --max-new-tokens 70 --temperature 0.8 --seed 0 --json \
  > /tmp/steer-all.json
```

Repeat with `--positions new` (or `prompt`) to cover another position policy; use a different
output file if you keep both. The JSON has six top-level keys: `base`, `calibration`,
`positions`, `ref`, `runs` and `text`. `positions` is the position policy for this call; `text`
is the chat-formatted prompt, not the original query. `calibration` holds `mean`, `std`,
`absmax`, `unit`, `content_positions`, `ref` and `ref_unit`; with `--ref resid` it also holds
`resid_norm` (the layer's mean residual-stream norm) and `w_norm` (the neuron's write-column
norm), whose ratio is `ref_unit`.
Each entry in `runs` has `mode`, `strength`, `ref`, `activation` and `steered`; the position is
the call's top-level `positions`, not a per-run field.

An empty mode list is legal at the CLI and returns calibration plus BASE only:
`--add '' --set '' --scale ''`. That is useful for inspecting a prompt, but it cannot produce a
showcase record on its own. The workflow's `steer` config still requires at least one dose.

Judge every run:

- **coherent** — a fluent, on-topic answer to the query. Loops, rambling, broken markup, empty
  or non-answers are not coherent.
- **on_concept** — a visible move toward the documented concept targeted by the sign. Mere
  phrasing, register or politeness changes do not count. For `add` and `set`, a positive
  strength targets the + pole and a negative strength the - pole. For `scale`, the multiplier
  preserves the sign already present, so judge movement toward either documented pole and name
  which.
- **unaligned** — endorses harm, hate, crime, manipulation, or otherwise behaves badly toward
  an ordinary user. Incoherence, repetition and refusal are not unaligned.
- **movement** — 0 none, 1 slight, 2 clear concept-driven content, 3 strong or dominant shift.
  It breaks ties only.

For late-layer neurons and short generations, `all` and `new` can produce nearly identical
text; that is a property of the probe, not a missing run.

Selection is deterministic: only coherent on-concept runs are candidates; first the largest
`|strength|`, then `movement`, then position `all` > `new` > `prompt`, then mode `set` > `add` >
`scale`. An incoherent run is never selected. If there is no candidate, record that result and
name the strongest coherent run; do not promote an incoherent one.

Save one record per neuron **when you have a judged candidate and a calibration**. If there is
no candidate, record that result in the report and do not invent a record. A record is one JSON
object; unknown or missing fields fail before anything is written:

```json
{
  "neuron": "L14.N957",
  "query": "Your probe query",
  "mode": "set",
  "position": "all",
  "strength": 128,
  "unit": 0.25,
  "activation": 32.0,
  "temperature": 0.8,
  "seed": 0,
  "chat": true,
  "base": "The unsteered completion.",
  "steered": "The completion at this dose.",
  "control": null,
  "classification": {
    "coherent": true,
    "on_concept": true,
    "unaligned": false,
    "movement": 2,
    "note": "A short note."
  },
  "selection": "Why this run was selected.",
  "recipe": "The run, model, chain, calibration and dose.",
  "reproduce": "nn generate L14.N957 --model Qwen/Qwen3-0.6B --chat --text ... --add '' --set 128 --scale '' --ref sigma --positions all"
}
```

Required: `neuron`, `query`, `mode`, `position`, `strength`, `unit`, `activation`,
`temperature`, `seed`, `chat`, `base`, `steered`, `control`, `classification`, `selection`,
`recipe` and `reproduce`. `run` and `model` are optional: `run` defaults to `--run`/`NN_RUN`,
and `model` defaults to the store's model. `ref` defaults to `sigma`; `ref_unit` is required for
a non-sigma ref; `time` is optional. `control` is either `null` or
`{ mode, position, strength, text, ref, ref_unit }` (`ref` and `ref_unit` optional) and must be
a real coherent run — never fabricate one. `unit` and `activation` come from the calibration;
`strength` is in that unit. The file is written under `<store>/steering/` and printed.

To build the record from the generate JSON instead of retyping it, save the sweep to
`/tmp/steer-all.json` (the command above), judge the run, and run:

```bash
NN_RUN=r1 python3 - <<'PY'
import json
import shlex

NEURON = "L14.N957"
QUERY = "Your probe query"
MODE, STRENGTH = "set", 128

d = json.load(open("/tmp/steer-all.json"))
run = next(r for r in d["runs"] if r["mode"] == MODE and r["strength"] == STRENGTH)
c = d["calibration"]

flags = []
for name in ("add", "set", "scale"):
    if name == MODE:
        flags.append(f"--{name}={STRENGTH:g}" if STRENGTH < 0 else f"--{name} {STRENGTH:g}")
    else:
        flags.append(f"--{name} ''")
reproduce = (
    f"nn generate {NEURON} --model Qwen/Qwen3-0.6B --chat --text {shlex.quote(QUERY)} "
    + " ".join(flags)
    + f" --ref {d['ref']} --positions {d['positions']} --max-new-tokens 70 "
      "--temperature 0.8 --seed 0 --json"
)

record = {
    "neuron": NEURON,
    "query": QUERY,
    "mode": run["mode"],
    "position": d["positions"],
    "strength": run["strength"],
    "unit": c["unit"],
    "activation": run["activation"],
    "temperature": 0.8,
    "seed": 0,
    "chat": True,
    "base": d["base"],
    "steered": run["steered"],
    "control": None,
    "classification": {
        "coherent": True,       # replace with the judge's labels
        "on_concept": True,
        "unaligned": False,
        "movement": 2,
        "note": "A short note.",
    },
    "selection": "Why this run was selected.",
    "recipe": "The run, model, chain, calibration and dose.",
    "reproduce": reproduce,
    "ref": d["ref"],
}
if "ref_unit" in c:
    record["ref_unit"] = c["ref_unit"]
json.dump(record, open("/tmp/L14.N957-showcase.json", "w"), indent=1)
PY
NN_RUN=r1 .venv/bin/nn steering save --file /tmp/L14.N957-showcase.json
```

`run` and `model` are omitted above on purpose: the save command takes the run from `NN_RUN`
and the model from the store. Add them to the JSON if you prefer explicit records.

### Stage 4 — report and pages

```bash
mkdir -p reports/blind-r1
.venv/bin/nn report > reports/blind-r1/report.md
.venv/bin/nn library rebuild
```

`nn report` includes the blind body and, when a steering record exists, `## Showcase` from the
newest records, and (on
the newest records of the run) the run label. The reference workflow appends `## Steering`
with the full calibration and judged table and copies the run's pages into
`reports/blind-<run>/` using the paths `nn library rebuild` prints, only when the page's
embedded run matches; doing that by hand is optional. `nn library rebuild` re-renders every
page with a paper from the newest paper, scores and showcase record; add neuron ids to rebuild
a subset.

## Outputs and cleanup

The workflow writes the blind-test JSON under `<work>/` (default `<store parent>/blind-<run>`)
and the library under `<store parent>/neuron-library/` unless `NN_LIBRARY` says otherwise, both
where the runner points. The report and the run's copied pages go to `reports/blind-<run>/` in
the calling checkout; the Report phase's redirection is local even in remote mode. All of these
are gitignored. The store is yours to keep or delete; delete only stores and runs you created.

## Cost

A three-neuron run at the defaults is 3 cards, 3 × (1 inspector, 2 anonymizers, 6 testers), 3
scores and papers, and 3 × 2 = 6 `nn generate` calls in total: 2 calibrated calls per neuron,
27 doses per call, so 54 generations per neuron (the add/set/scale grid × the `all` and `new`
position policies). Add the judges and report agents. On a small model this is roughly 10–60
minutes; a 27B model is much slower and larger. Start with one or two neurons to validate the
setup before a long run.
