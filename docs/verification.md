# Verification

How to check a checkout: static checks, the automated test suite, the repository's CI
configuration and the repository hygiene checks. Run the commands from the repository root on a
clean checkout with no experiment artifacts in the tree. For the end-to-end smoke run on a
machine with a GPU, see [manual-smoke.md](manual-smoke.md).

## 1. Static checks

```bash
python3 --version                                   # 3.10 or newer
PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/nn-pyc" python3 -m py_compile nn/*.py tests/*.py examples/*.py
bash -n scripts/nn.sh
bash -n run.example/remote.sh.example
python3 -c 'import json; json.load(open(".dsh/workflows/blind-neuron-test.meta.json"))'
```

Compile the workflow body as the DSH engine would (the script is a body, not an ES module):

```bash
node -e '
const fs = require("fs");
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
new AsyncFunction("agent","phase","log","parallel","args",
  fs.readFileSync(".dsh/workflows/blind-neuron-test.js", "utf8"));
console.log("workflow body compiles");
'
```

The tracked tree should also contain no caches or stray artifacts:

```bash
find . -name __pycache__ -o -name "*.pyc" -o -name ".DS_Store"
```

## 2. Automated tests

```bash
python3 -m unittest discover -s tests -v
```

Expected: `Ran 48 tests ... OK` (the count can grow; it must be `OK`). The suite uses
`unittest` and numpy only, never torch, transformers or datasets, so it runs on a CPU-only
machine. It covers:

- `tests/test_library_scope.py` — the run-scoped library rule: a rebuild of one run never
  removes another run's winning page, a run removes its own stale page, a page without
  readable metadata is never removed, and the CLI wiring is correct.
- `tests/test_showcase.py` — strict validation and Markdown rendering of showcase records.
- `tests/test_cli_helpers.py` — Fisher scoring, formatting, dose/verdict helpers, contrast
  pairing, no model or store needed.
- `tests/test_docs_and_workflow.py` — relative links in the three top-level docs, the runner
  and `run.example/` template files, the meta's ten stage titles, the `phase()` calls and
  agent phase labels, and the configurable-runner wiring. It is a structural check, not a run.
- `tests/test_no_machine_leaks.py` — the machine-leak guard over the tree (skipping `tests/`
  and operational directories) and the `.gitignore` coverage of run, report and store dirs.

The suite has no subprocess and never loads a model, so the runner and workflow are checked as
files, not executions. For a GPU/host smoke run, follow [manual-smoke.md](manual-smoke.md); the
full three-stage pipeline is described in [workflow.md](workflow.md).

## 3. CI

`.github/workflows/tests.yml` runs the same checks on every push and pull request: dependency
installation, `nn --help`, `py_compile`, `bash -n`, the workflow-body compile check and the
dependency-free suite, on Python 3.10, 3.11 and 3.12.

## 4. Repository hygiene

The tracked tree should contain no machine-specific paths, credentials or generated artifacts:

```bash
git grep -n -I -E '/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+' -- . ':!tests/' ':!docs/verification.md'
git grep -n -I -E 'ssh +[A-Za-z0-9._-]+@|rsync +.*[A-Za-z0-9._-]+:' -- . ':!tests/' ':!docs/verification.md'
git grep -n -I -E 'BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|ssh-(rsa|ed25519)|api[_-]?key[[:space:]]*[=:]|password[[:space:]]*[=:]|secret[_-]?key[[:space:]]*[=:]' -- . ':!tests/' ':!docs/verification.md'
find . -maxdepth 3 \( -name reports -o -name runs -o -name run -o -name neuron-library \
  -o -name .venv -o -name __pycache__ \) -print
```

Expected: the home and secrets patterns print nothing; the ssh/rsync pattern prints only the
documented public placeholders in `docs/remote.md`; the find prints nothing on a clean checkout.
A local setup may legitimately have the gitignored `run/` and `.venv/`; they are never committed.
