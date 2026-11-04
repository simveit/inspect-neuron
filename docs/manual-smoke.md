# Manual smoke checklist

Use this after changing the workflow, the CLI or the templates. It is deliberately small: a few
minutes of commands on a laptop or a single GPU, not a full experiment.

Prerequisites: Python 3.10+, the repository root, and (for the server step) a GPU with enough
VRAM for `Qwen/Qwen3-0.6B`. The model download is the slow part on a first run; set `HF_HOME`
to a disk with room.

## 1. Install and static checks

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/nn --help
./scripts/nn.sh --check
./scripts/nn.sh --help
python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/nn-pyc" python3 -m py_compile nn/*.py tests/*.py examples/*.py
bash -n scripts/nn.sh
bash -n run.example/remote.sh.example
```

- [ ] `nn --help` lists the subcommands; the tests print `OK`.
- [ ] `./scripts/nn.sh --check` prints the local `nn`, Python, store status, GPU and disk.
- [ ] `./scripts/nn.sh --cmd 'echo "$NN_STORE"'` prints an empty line (no store set yet), so the
  local `--cmd` path works.
- [ ] `NN_RUNNER=/bin/echo ./scripts/nn.sh top L1.N0 --pole +` prints the arguments unchanged
  (the explicit runner override works).

## 2. Store from a local corpus (CPU is fine)

```bash
export NN_STORE="$PWD/runs/smoke-store"
.venv/bin/nn harvest --device cpu --model Qwen/Qwen3-0.6B \
  --texts examples/tiny-corpus.jsonl --tokens 50000
ls "$NN_STORE"          # meta.json vals.npy windows.npy fires.npy tokenizer.json
```

- [ ] The harvest finishes and the store files exist.
- [ ] The store is a full all-neuron store (about 0.9 GB for Qwen3-0.6B even from 60 documents);
  a targeted `--neurons` store is about 13 MB.
- [ ] Rerun the same command: it resumes (or reports that the corpus is exhausted) without
  rebuilding from scratch.
- [ ] `nn select 1 --seed 0` prints one neuron id; `nn select 1 --seed 0 --json` prints JSON.
- [ ] `nn top <neuron> --pole + -n 3` prints corpus examples.
- [ ] `nn library rebuild` on this store says there is no paper (or renders nothing) without
  crashing.

Delete the smoke store when done if you do not need it:

```bash
rm -rf "$NN_STORE"      # only the store you just created
```

## 3. Server and model tools (GPU)

```bash
.venv/bin/nn serve
# another shell:
curl -fsS http://127.0.0.1:8765/info
```

- [ ] `/info` names the store's model.
- [ ] `nn top`, `nn run`, `nn steer`, `nn ablate` and `nn attrib` answer for a real neuron from
  the store.
- [ ] `nn generate <neuron> --chat --text "..." --add 8,-8 --set 16,-16 --scale 2 --ref sigma
  --positions all --max-new-tokens 20 --json` returns a BASE and five runs, with calibration.
- [ ] Stop only the server you started, using its pid.

## 4. Manual blind path (one neuron)

Use a neuron from the store and a text that plausibly fires it. This checks the record
plumbing; it is not a scientific result.

```bash
.venv/bin/nn card submit <neuron> --core "A test explanation." --negative "" \
  --offshoots "" --residue "Smoke test."
# replace <card-id> with the id nn printed
.venv/bin/nn card redact <card-id> --pole + --text "A short abstract description."
.venv/bin/nn card read <card-id> --pole +
cat > /tmp/smoke-tests.json <<'EOF'
[{"card": "<card-id>", "pole": "+", "tester": "smoke", "run": "smoke",
  "blind_max": "A text that should fire it.",
  "new": ["A related text.", "Another related text."],
  "near_miss": ["An unrelated text.", "Another unrelated text."],
  "distant": ["A distant text."]}]
EOF
.venv/bin/nn score /tmp/smoke-tests.json
.venv/bin/nn paper save <neuron> --title "Smoke neuron" --summary "A test paper." \
  --clean 2 --clean-why "Plumbing only." --interesting 1 --interesting-why "Smoke test."
.venv/bin/nn report | head -20
.venv/bin/nn library rebuild <neuron>
```

- [ ] `nn score` writes a result and prints a verdict; with several tests it also writes a
  pooled record.
- [ ] `nn paper save` prints a library line; a page exists under `neuron-library/` only if a
  pole won.
- [ ] `nn report` shows the card, the score and the paper.
- [ ] `nn library rebuild <neuron>` re-renders the page and the index without a server.

## 5. Workflow (DSH, one neuron)

The reference path needs an existing DeepSeek Harness (DSH) session; without one, use the manual
path in section 4. With a real store and a running server, launch the workflow with a minimal
probe:

```js
{
  run: 'smoke',
  store: '<absolute store path>',
  select: { n: 1, seed: 0 },
  testers: 1,
  inspectors: 1,
  paper: true,
  steer: { add: [8, -8], set: [16, -16], scale: [2], positions: ['all'],
           max_new_tokens: 20 },
}
```

- [ ] The Screen stage is skipped when `screen` is absent; Setup resolves one neuron.
- [ ] Inspect, Anonymize, Test, Score and Paper run; a card, a score and a paper exist.
- [ ] Steer judges every expected run and saves one showcase record when a candidate and
  calibration exist; otherwise it records that none was saved.
- [ ] `reports/blind-smoke/report.md` is non-empty and has `## Showcase` when a steering record
  was saved; with the DSH workflow
  it also has `## Steering`.
- [ ] The run's `<neuron>.html` is copied when the neuron has a page.
- [ ] The workflow returns `ok: true` (per-neuron failures are listed but do not crash the run).

## 6. Runner and remote plumbing (no remote host required)

```bash
bash -n run.example/remote.sh.example
if [ ! -e run/remote.sh ]; then
  mkdir -p run
  cp run.example/remote.sh.example run/remote.sh
  chmod +x run/remote.sh
  # With no run/run.env, --check must fail with a clear "set NN_REMOTE_HOST" message:
  ./scripts/nn.sh --check || true
  rm -f run/remote.sh
  rmdir run 2>/dev/null || true
else
  echo "run/remote.sh already exists; skipping the template check"
fi
```

- [ ] The missing-host error is clear and non-destructive.
- [ ] `run/` is ignored by git (`git check-ignore run/remote.sh`).
- [ ] With a real remote host, `./scripts/nn.sh --check`, `--cmd`, a synchronous call and
  `--detach` all work (see [remote.md](remote.md)).

## 7. Repository hygiene

- [ ] `python3 -m unittest discover -s tests -v` still passes.
- [ ] The hygiene checks in [verification.md](verification.md#4-repository-hygiene) return
  nothing unexpected.
- [ ] `git status --short` shows no reports, runs, stores, logs, caches or `run/` files.
- [ ] A fresh clone (or copy) installs and passes the tests with no reliance on your shell
  history, aliases or pre-existing environment variables.
