# run/: machine-local configuration (gitignored)

`run/` is gitignored because it holds machine-specific values such as a host alias, a home
directory or a private path. The committed templates in `run.example/` are examples; copy them
into `run/` and keep your filled-in copies there.

## Local mode (default)

Do nothing. `./scripts/nn.sh` runs `nn` on this machine, preferring `.venv/bin/nn` and falling
back to `nn` on `PATH`. Every argument, `--cmd` included, goes straight to `nn` or to the shell.
`./scripts/nn.sh --check` reports the local `nn`, Python, store status, GPU and disk. Set
`NN_LOCAL=1` to ignore a `run/remote.sh` that exists, or `NN_BIN=/path/to/nn` to choose the
local executable.

## Remote GPU host (optional)

Copy the two templates into `run/` and fill in your own values:

```bash
mkdir -p run
cp run.example/remote.sh.example run/remote.sh
cp run.example/run.env.example run/run.env
chmod +x run/remote.sh
$EDITOR run/run.env        # NN_REMOTE_HOST, NN_REMOTE_REPO
```

Once `run/remote.sh` exists, `./scripts/nn.sh` delegates every call to it unchanged:

```bash
./scripts/nn.sh --check                          # ssh, repo, nn and GPUs on the host
./scripts/nn.sh --cmd 'cat "$NN_STORE/meta.json"'
./scripts/nn.sh --dry-run top L14.N957 --pole +  # print what would run on the host; no ssh
CUDA_VISIBLE_DEVICES=<gpu> ./scripts/nn.sh harvest --tokens 2e7
./scripts/nn.sh --detach serve                   # remote only: prints the pid and remote log
```

`run/remote.sh` forwards `NN_STORE`, `NN_SERVER`, `NN_RUN`, `NN_LIBRARY`,
`CUDA_VISIBLE_DEVICES` and `HF_HOME` for every call.

## Precedence

1. `NN_RUNNER=/path/to/runner ./scripts/nn.sh ...` uses that runner for everything.
2. `run/remote.sh` if it exists and `NN_LOCAL` is not `1`.
3. `nn` locally (or `NN_BIN`).

In committed files, write generic placeholders (`YOUR_HOST`, `/path/to/repo`, `user@host`)
instead of real paths or host names.
