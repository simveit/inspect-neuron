# Tests

Dependency-free unit tests (stdlib `unittest`; only numpy is needed, no GPU):

```bash
python -m unittest discover -s tests -v
```

The 48 tests cover pure CLI helpers and score math, the run-scoped library rule, showcase
record validation and Markdown rendering, the docs/workflow structural contract, and a
machine-leak guard. They do not execute the runner (`scripts/nn.sh`), start a server or load a
model; [docs/manual-smoke.md](../docs/manual-smoke.md) is the end-to-end smoke check and
[docs/verification.md](../docs/verification.md) covers the static checks and the test suite.
