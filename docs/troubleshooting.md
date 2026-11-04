# Troubleshooting

Work down from the first failure. The commands assume the repository root; use
`./scripts/nn.sh` instead of `.venv/bin/nn` if you want the local/remote runner to decide where
`nn` runs. Paths shown as `/path/to/...` are placeholders.

## Install and import

**`nn: command not found`**

The venv is not active and the package is not installed globally. Use the venv binary or the
runner:

```bash
.venv/bin/nn --help
./scripts/nn.sh --help
```

If `./scripts/nn.sh` also fails, check that `.venv/bin/nn` exists or that `nn` is on `PATH`
inside the environment where you run the command. Reinstall with:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

**`ModuleNotFoundError: No module named 'torch'` (or `transformers`, `datasets`)**

Only the commands that run the model import those packages. Install the dependencies in the
active environment:

```bash
.venv/bin/pip install -e .
```

Tests need only numpy. `python3 -m unittest discover -s tests -v` is expected to work without
torch installed.

**`ImportError` for `nn` when running from another directory**

Run `pip install -e .`, or set `PYTHONPATH=/path/to/repo`. The tests insert the repository root
themselves.

**`python3: command not found` / wrong Python**

Use Python 3.10 or newer (`python3 --version`) and rebuild the venv with that interpreter.

## Store

**`nn: no store: set NN_STORE or pass --store`**

Every store command needs a store. Export `NN_STORE` (absolute path is safest) or pass
`--store /path/to/store`. The workflow does not read `NN_STORE` for its own config: pass
`store` in its args.

**`nn: [Errno 2] No such file or directory: '.../meta.json'`**

The store is missing or unfinished. Run the harvest to completion (or resume it) before
`nn top`, `nn paper`, `nn library rebuild`, etc. A store is a directory with `meta.json`,
`vals.npy`, `windows.npy`, `fires.npy` and `tokenizer.json`.

**Harvest stopped early / no examples for a neuron**

A local corpus may be shorter than `--tokens`; that is fine. If the process was interrupted,
rerun the same command: it resumes from its checkpoint. A resume must use the same model,
settings and corpus; to change them, use a new store directory.

**`neurons lists the same neuron twice` / targeted-store resume fails**

Deduplicate the list and reuse exactly the same `--neurons` list when resuming a targeted
store. A targeted store cannot be extended into a full store (or the reverse); build a new
directory instead.

**`only N live neurons to choose from`**

`nn select` samples live neurons; a tiny or narrow store may have fewer than requested. Lower
`n`, raise `--top` (or use `--top` with a larger value), or use an explicit list.

## Server and model

**`no nn server at ...` in Setup, or `connection refused`**

Start `nn serve` yourself, with the same store the workflow uses:

```bash
NN_STORE=/path/to/store .venv/bin/nn serve
curl -fsS "$NN_SERVER/info"
```

The model comes from the store's `meta.json` unless `--model` is given. The workflow never
starts or stops the server.

**`model ... does not match` / Setup reports a mismatch**

The store was harvested with one model and the server loaded another. Restart the server
without `--model` (so it reads the store), or point the workflow at the store that matches the
running server. Check both:

```bash
cat "$NN_STORE/meta.json"
curl -fsS "$NN_SERVER/info"
```

**Port 8765 is in use**

Either another server is already running (use it if it serves your store), or choose a
different URL and keep it consistent everywhere:

```bash
NN_SERVER=http://127.0.0.1:8766 .venv/bin/nn serve
NN_SERVER=http://127.0.0.1:8766 ./scripts/nn.sh --cmd 'curl -fsS "$NN_SERVER/info"'
```

Pass the same `server` in the workflow args.

**CUDA not available / `no CUDA GPUs are available`**

Check the driver and visibility (`nvidia-smi`, `echo "$CUDA_VISIBLE_DEVICES"`). On a shared host
do not change another user's process; use a free device. For a quick correctness check use
`--device cpu` (slow), especially with the tiny corpus.

**CUDA out of memory**

The model, the sequence length, the batch and the generation grid all use memory. Try, in
order:

1. a smaller `--max-new-tokens` and fewer strengths/positions in the probe;
2. a smaller model (the examples use `Qwen/Qwen3-0.6B`);
3. a targeted store (`nn harvest --neurons`) instead of a full one;
4. lower `--batch`/`--ctx` at harvest, or `--max-docs` on a test corpus;
5. a GPU with more free memory.

There is no hidden fallback to CPU; pass `--device cpu` explicitly if you want it.

**A server or harvest is left running**

Only stop a process you started. If you used the remote template's `--detach`, it printed the
pid and log path; use those. On a local machine, `kill <pid>` where `<pid>` is your process.
Never kill, restart or attach to a process you did not start.

## Datasets and corpora

**Hugging Face authentication or a private/gated dataset**

Log in with the current Hugging Face CLI (`hf auth login`; older installs use
`huggingface-cli login`) or set `HF_TOKEN` in the environment where `nn` runs. Accept the
dataset's terms on its page. A local `.jsonl` file is the simplest way to avoid authentication
entirely.

**`field 'text' of ... is not a string column`**

The column name is wrong or the values are not strings. Inspect the dataset schema and pass the
right `--field` (comma-separated for several columns; they are joined with newlines). For a
JSONL file, each object needs the named field, or use a plain-text file.

**`--where` keeps nothing**

Matching is case-insensitive string equality on the raw column value. A boolean column may be
`true`/`false` rather than `True`/`False`; check the dataset viewer. To keep everything, omit
`--where`. For the intent screen, the positive and benign sources must each yield at least two
usable texts.

**`nn: need at least 2 texts per class, got N positive and M negative (try a larger --max-texts or a lower --tokens)`**

Increase `--max-texts`, lower `--tokens` (texts shorter than that are skipped), or loosen the
`--match`/`--where` filters. The screen reads equal-length prefixes and needs at least three
tokens per text.

**FineWeb seems to download forever**

FineWeb is streamed, not downloaded as a corpus; the first run still downloads/loads the model
and warms the dataset stream. Use a local corpus and `--max-docs` for a quick smoke test, and
keep `HF_HOME` on a disk with room.

## Workflow

**The workflow tool is not a command**

DSH has no name-based launcher. The Lead (the lead agent of an existing DSH session) must read
both files and call the `workflow` tool with `meta`, `script` and `args`; see
[workflow.md](workflow.md#launch-through-dsh). Asking for a shell command named after the
workflow will not work.

**`store is required: pass store: '/path/to/store'`**

The workflow deliberately has no machine-specific default. Add `store` to the args. Setting
`NN_STORE` in the shell is enough for `nn` commands but does not reach the workflow config.

**`runner` not found / commands run in the wrong place**

The default `runner: './scripts/nn.sh'` is resolved from the checkout root. If you pass a
custom runner, make it executable and able to run every `nn` argument plus `--cmd`. Test it
before a long run:

```bash
./scripts/nn.sh --cmd 'pwd; echo "$NN_STORE"; ls "$NN_STORE/meta.json"'
```

**Setup stops with `no nn server` or a store/model mismatch**

See the server section above. The Setup agent checks the server, the store's `meta.json` and
the neuron list before any expensive work. Fix the first failure rather than rerunning.

**The screen stops with a problem**

Read the returned `problem`: too few usable neurons, no texts after filtering, a harvest
failure, or a failed `nn top` check. Loosen the screen filters (`max_texts`, `tokens`, `match`,
`where`), lower `picks`, or harvest a full store and use `select`/`neurons` instead.

**Paths in args look right locally but fail remotely**

All workflow paths are evaluated where the runner runs. Use absolute paths, and remember that
`store`, `screen.store`, `work` and the server URL have that perspective. See
[remote.md](remote.md).

**The run wrote nothing to `reports/`**

The Report phase writes `reports/blind-<run>/` in the calling checkout; the `mkdir` and output
redirection are local even in remote mode. Check the run label (`NN_RUN`/`run`), the `reports`
config, and whether the process could create the folder. `nn report` needs saved score records
for its per-pole sections; with no scored neuron it still writes the header and summary, and a
neuron that only has a saved paper still gets a paper section.

**The report has no neuron sections or hides an earlier run**

`NN_RUN`/`--run` limits the report to that run. Omit the run label to see the newest records
across runs, and make sure the cards, scores and papers were written with the run label you
are reading.

**A library page disappeared after rebuilding another run**

A run-scoped `nn library rebuild` removes a page only if that run owns it or no run still has a
winning score for the neuron. A page without readable embedded metadata is never removed. If
you see a deletion across runs with this revision, capture the store's scores and the exact
command in an issue.

**No HTML page for a neuron**

Pages exist only for a winning pole (newest score = WIN). Ties and losses stay in the store by
design. `nn paper <neuron>` shows the scores; `nn library rebuild <neuron>` re-renders from the
store.

**Agents write into the checkout unexpectedly**

The workflow instructs agents to keep scratch under `/tmp` and to write only under
`reports/blind-<run>/` in the Report phase. If a harness or agent ignores that, check
`git status` and remove only the files you created. The store and the report folder are
gitignored.

## Tests and verification

**`unittest` reports an import error**

Run from the repository root: `python3 -m unittest discover -s tests -v`. The tests use
`tests.helpers` and require numpy, not torch.

**A docs/workflow test fails after editing**

The suite checks that relative links in `README.md`, `AGENTS.md` and `DESIGN_DOC.md` resolve,
that the workflow's ten meta stage titles, `phase()` calls and agent phase labels agree, and
that its commands use the configurable runner. Fix the docs or the workflow rather than the
test.

**Hygiene checks fail**

Run the commands in [verification.md](verification.md#4-repository-hygiene). Any tracked file
mentioning a real host alias, a home directory, an internal store path or a private SSH alias is
a bug.

## Remote mode

See [remote.md](remote.md#troubleshooting). The short version: fill in `run/run.env`, check
`./scripts/nn.sh --check`, keep every path absolute, start the server on the remote host, and
remember that `--detach` logs live under the remote checkout's `run/logs/`.
