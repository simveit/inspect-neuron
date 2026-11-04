# Optional: run `nn` on a remote GPU host

**Local mode is the default.** On a machine with a GPU, run `nn` (or `./scripts/nn.sh`) in this
checkout; nothing else is required. This page is for the case where the checkout is on a laptop
and the GPU is on another machine you reach over SSH. Neuron ids in the examples (for example
`L14.N957`) are illustrative.

The design is deliberately small: one committed runner (`scripts/nn.sh`) and one optional,
machine-local script (`run/remote.sh`). Host names, home directories and store paths live only
in the gitignored `run/` directory. Never commit them.

## How it works

```
./scripts/nn.sh ARGS
   ├─ NN_RUNNER set?                    -> exec that runner with ARGS
   ├─ run/remote.sh exists, NN_LOCAL≠1? -> exec run/remote.sh with ARGS (including --cmd/--detach)
   └─ otherwise                         -> .venv/bin/nn or nn on PATH (or NN_BIN)
```

`run/remote.sh` is a template you fill in. It must accept the same calls `nn` accepts, plus:

| Call | Meaning |
|---|---|
| `run/remote.sh ARGS` | run `nn ARGS` in the remote checkout |
| `run/remote.sh --cmd '<command>'` | run a shell command in the remote checkout, with the `NN_*` environment |
| `run/remote.sh --detach ARGS` | start a long `nn` command in the background and print its pid and log path |
| `run/remote.sh --dry-run ARGS` | print the command that would run on the host, without SSH |
| `run/remote.sh --check` | print the host, repo, remote `$HOME`, `nn --help` status and GPUs |

The supplied template in [../run.example/remote.sh.example](../run.example/remote.sh.example)
implements these calls over `ssh` (`--dry-run` only prints), forwards `NN_STORE`, `NN_SERVER`,
`NN_RUN`, `NN_LIBRARY`, `CUDA_VISIBLE_DEVICES` and `HF_HOME`, and writes `--detach` logs under
`$NN_REMOTE_REPO/run/logs/`. Read it before using it; it is short on purpose.

## Setup

```bash
mkdir -p run
cp run.example/remote.sh.example run/remote.sh
cp run.example/run.env.example run/run.env
chmod +x run/remote.sh
$EDITOR run/run.env
```

`run/run.env` needs at least:

```bash
NN_REMOTE_HOST=user@host          # or an SSH alias from ~/.ssh/config
NN_REMOTE_REPO=/path/to/repo      # absolute path of this checkout on the host
# NN_REMOTE_NN=.venv/bin/nn       # optional, relative to NN_REMOTE_REPO or absolute
```

Both files are ignored by git (`run/` is in `.gitignore`). Keep SSH keys in your normal SSH
agent/config, not in this repository. The template never stores a password.

### Populate the host checkout

The runner executes `nn` from `NN_REMOTE_REPO`, so that directory must contain the checkout on
the host (the commands below create it) — the checkout there, not the calling one, is the code
that runs. From the machine that has the checkout (replace `YOUR_HOST` and the path with your own
values):

```bash
git archive --format=tar.gz -o /tmp/inspect-neuron.tar.gz HEAD
scp /tmp/inspect-neuron.tar.gz YOUR_HOST:/tmp/
ssh YOUR_HOST 'mkdir -p /path/to/repo && tar -xzf /tmp/inspect-neuron.tar.gz -C /path/to/repo'

# install there with the host's Python 3.10+:
ssh YOUR_HOST 'cd /path/to/repo && python3 -m venv .venv && .venv/bin/pip install -e .'
```

`NN_REMOTE_NN` defaults to `.venv/bin/nn` relative to `NN_REMOTE_REPO`, so a venv inside the
checkout needs no further configuration. The host install needs network access; if pip selects a
CPU-only torch wheel, install the CUDA wheel matching the host's CUDA version from
<https://pytorch.org/get-started/locally/> first and re-run the editable install (the same step as
the local [install](../README.md#install)). The host copy is what every `nn` command uses, so
re-sync it after committing changes (`git archive HEAD` ships committed state only — use `rsync`
for a dirty tree). A re-extract overwrites tracked files but does not delete stale ones, so use a
fresh directory or `rsync --delete` if you need a clean sync.

Check the connection and the remote environment:

```bash
./scripts/nn.sh --check
```

This prints the host, repo, remote `$HOME`, whether `nn --help` succeeds, and `nvidia-smi`
output when available. It does not start or stop anything. In local mode (no `run/remote.sh`,
or `NN_LOCAL=1`), `./scripts/nn.sh --check` reports the local `nn`, Python, store, GPU and disk
instead.

## Running

```bash
# A quick command where nn runs (the same environment as nn):
./scripts/nn.sh --cmd 'cat "$NN_STORE/meta.json"'
./scripts/nn.sh --cmd 'curl -fsS "$NN_SERVER/info"'

# Synchronous nn commands:
./scripts/nn.sh top L14.N957 --pole + -n 20
./scripts/nn.sh select 3 --seed 0

# Long commands: start them in the background; the template prints pid and log path.
NN_STORE=/path/to/store ./scripts/nn.sh --detach harvest --model Qwen/Qwen3-0.6B --tokens 2e7
NN_STORE=/path/to/store ./scripts/nn.sh --detach serve
```

The `--detach` log path is on the remote host. Follow it with a command you run yourself, for
example `ssh "$NN_REMOTE_HOST" 'tail -f /path/to/repo/run/logs/<file>.log'` (or the template's
printed `log:` line). Only stop a process you started:

```bash
ssh "$NN_REMOTE_HOST" 'kill <pid>'    # only your own nn serve or harvest
```

Never kill, restart or attach to a process you did not start. GPUs are often shared; check
before a long job:

```bash
./scripts/nn.sh --cmd 'nvidia-smi'
```

Pick a free device and pin it with `CUDA_VISIBLE_DEVICES=YOUR_GPU` on the command, or pass `gpu:
<index>` in the workflow args (that pins screen/harvest/generate; the server stays where it is).
There is no repository-specific GPU picker; use `nvidia-smi` and your host's scheduler.

## The server from the control plane

Start `nn serve` where `nn` runs (on the GPU host), with the store whose `meta.json` names the
model:

```bash
NN_STORE=/path/to/store ./scripts/nn.sh --detach serve
./scripts/nn.sh --cmd 'curl -fsS "$NN_SERVER/info"'
```

`NN_SERVER` in `run/run.env` or in the calling environment is the URL **as seen from the GPU
host**, typically `http://127.0.0.1:8765`. The workflow's Setup agent checks it through
`--cmd`, so it runs on the same side as the server. If you need to reach the server from your
laptop for debugging, use an SSH tunnel; do not expose it publicly.

The workflow does not start or stop the server. On the GPU host, a server can serve several
runs; stop only the one you started when you are done.

## Results

The store, the work directory and the neuron library are written on the GPU host. The report
and the pages copied for the run are written in the **calling** checkout, because the Report
phase's `mkdir` and output redirection run locally even when `nn` itself runs remotely. Pull
the remote library or logs when you want them:

```bash
rsync -a "$NN_REMOTE_HOST:$NN_REMOTE_REPO/run/logs/" run/logs/
```

The destination is gitignored. If you also want the store locally, pull it and re-render the
report and the library without a server or GPU:

```bash
rsync -a "$NN_REMOTE_HOST:$NN_REMOTE_REPO/runs/my-store/" runs/my-store/
NN_STORE="$PWD/runs/my-store" NN_RUN=r1 .venv/bin/nn report > reports/blind-r1/report.md
NN_STORE="$PWD/runs/my-store" .venv/bin/nn library rebuild
```

`nn report` and `nn library rebuild` need no server and no GPU. `nn library rebuild` writes
the library beside the local store unless `NN_LIBRARY` points elsewhere.

## Workflow with a remote runner

Set `runner: './scripts/nn.sh'` (the default) in the workflow args. Because `run/remote.sh`
exists, every `nn` call and every `--cmd` is delegated to the host. Keep `store` absolute and
remote-side; use the remote host's `NN_SERVER` value. The workflow's `--cmd` writes the screen
and picks files under `/tmp` and the blind-test JSON under the work directory on the remote
host, which is where the agents expect it. The report and the run's pages are still written
into `reports/blind-<run>/` in the calling checkout; only the store and the library stay on the
GPU host. If you want everything local, omit `run/remote.sh` and run the workflow against a
local model; that is a different tradeoff.

## Troubleshooting

| Symptom | Check |
|---|---|
| `set NN_REMOTE_HOST` error | `run/run.env` exists and is filled in; you copied it from `run.example/` |
| `Permission denied (publickey)` | your normal SSH config/agent works for `ssh user@host`; no keys belong in this repo |
| `nn: command not found` remotely | `NN_REMOTE_NN` points at the remote venv, or run `pip install -e .` there |
| `no such file` for the repo | `NN_REMOTE_REPO` is an absolute path on the **remote** host |
| server not answering | start it remotely; `NN_SERVER` is remote-side; check `curl` through `--cmd` |
| paths behave differently | all paths in workflow args are evaluated where the runner runs; use absolute paths |
| no logs | `--detach` writes under `$NN_REMOTE_REPO/run/logs/`; the directory is created for you |
| GPU busy / OOM | `nvidia-smi`, pick another device, reduce batch/context/tokens, or use a targeted store |

The general list is [troubleshooting.md](troubleshooting.md).
