#!/usr/bin/env bash
# The one runner every workflow command uses. It runs nn locally by default and delegates
# every argument (--cmd included) to the machine-local run/remote.sh when that file exists.
#
#   ./scripts/nn.sh <nn args>              run nn from this checkout (prefers .venv/bin/nn)
#   ./scripts/nn.sh --cmd '<command>'      run a shell command where nn runs, same env, stdin kept
#   ./scripts/nn.sh --check                show nn/python/store/GPU/disk status
#   ./scripts/nn.sh --detach <nn args>     start nn in the background; prints pid and log path
#   ./scripts/nn.sh --dry-run <nn args>    print the command instead of running it
#   ./scripts/nn.sh --help                 show this help
#
# Machine-local settings: copy run.example/ to run/ (gitignored); NN_RUNNER overrides the runner,
# NN_LOCAL=1 forces local, NN_BIN picks the nn executable, NN_LOGS moves the detached logs.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$ROOT/run/env.sh" ] && . "$ROOT/run/env.sh"
REMOTE=${NN_REMOTE:-$ROOT/run/remote.sh}

resolve_nn() {
  if [ -n "${NN_BIN:-}" ]; then printf '%s' "$NN_BIN"; return; fi
  if [ -x "$ROOT/.venv/bin/nn" ]; then printf '%s' "$ROOT/.venv/bin/nn"; return; fi
  command -v nn
}

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
}

has_remote() { [ "${NN_LOCAL:-0}" != 1 ] && [ -e "$REMOTE" ]; }
delegate() { if [ -x "$REMOTE" ]; then exec "$REMOTE" "$@"; fi; exec bash "$REMOTE" "$@"; }

if [ -n "${NN_RUNNER:-}" ]; then exec "$NN_RUNNER" "$@"; fi

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --cmd)
    shift
    [ "$#" -ge 1 ] || { echo 'scripts/nn.sh: --cmd needs a command' >&2; exit 2; }
    if has_remote; then delegate --cmd "$*"; fi
    exec bash -c "$*" ;;
  --check)
    shift
    if has_remote; then delegate --check "$@"; fi
    nn=$(resolve_nn || true)
    if [ -n "$nn" ]; then echo "nn: $nn"; else echo "nn: NOT FOUND (python3 -m venv .venv && .venv/bin/pip install -e .)"; fi
    py=python3
    if [ -n "$nn" ] && [ -x "$(dirname "$nn")/python" ]; then py=$(dirname "$nn")/python; fi
    "$py" -c 'import sys; print("python:", sys.version.split()[0])' 2>/dev/null || true
    if [ -n "${NN_STORE:-}" ]; then
      if [ -f "$NN_STORE/meta.json" ]; then echo "store: $NN_STORE"; else echo "store: $NN_STORE (missing meta.json)"; fi
    else
      echo "store: NN_STORE is not set"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -8
    else
      echo "gpu: nvidia-smi not found (CPU-only works but is slow)"
    fi
    df -h "$ROOT" 2>/dev/null | tail -1 || true
    exit 0 ;;
  --detach)
    shift
    if has_remote; then delegate --detach "$@"; fi
    logdir=${NN_LOGS:-$ROOT/run/logs}
    mkdir -p "$logdir"
    tag=$(printf '%s' "${1:-nn}" | tr -cd '[:alnum:]_.-')
    log="$logdir/$(date +%Y%m%d-%H%M%S)-${tag:-nn}.log"
    nohup "$(resolve_nn)" "$@" >"$log" 2>&1 </dev/null &
    echo "started nn (pid $!)"
    echo "log: $log"
    exit 0 ;;
  --dry-run)
    shift
    if has_remote; then delegate --dry-run "$@"; fi
    echo "nn $(resolve_nn) $*"
    exit 0 ;;
  --) shift ;;
esac

if has_remote; then delegate "$@"; fi
exec "$(resolve_nn)" "$@"
