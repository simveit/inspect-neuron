"""Stream a text corpus through the model and keep each neuron's strongest examples.

By default the corpus is FineWeb (sample-10BT), but any Hugging Face dataset or
JSONL/plain-text file can be used instead (see nn.textsource and the
--dataset/--texts options of `nn harvest`); the source is stored in meta.json
and must match when resuming.

Documents are tokenized, each prefixed with the separator token, and packed into
`ctx`-token sequences. For every neuron, regime and sequence, only the sequence's
peak token is a candidate (so one long document can't fill a list), and position
0 is skipped. Candidates merge into a running top-k on the GPU after every layer.
Only the store below is written; text and full activations are never kept.

Pass `neurons=[...]` to build a targeted store for just those neurons (use
`nn screen` or `nn rank` to find them): only their channels of h and gate are
indexed per layer, their rows of vals/windows are compact, and fires.npy stays
dense so nn.store.Store.alive()/select still works. The same list is required
to resume a targeted store, and a targeted store cannot be extended into a full
one (or the reverse) -- use a new directory.

Store (one directory):
  meta.json       model, settings, tokens seen, stream position for resuming;
                  "neurons": [[layer, idx], ...] in a targeted store
  vals.npy        float32 [L, I, 4, k]     peak values per neuron and regime, descending
                  (targeted: [S, 4, k], one row per tracked neuron)
  windows.npy     int32   [L, I, 4, k, W]  token ids around each peak (peak at index
                                           `before`, -1 = padding)
                  (targeted: [S, 4, k, W])
  fires.npy       int64   [L, I, 2]        tokens with h > fire (pole +), h < -fire (pole -);
                                           dense in both layouts
  tokenizer.json  for decoding windows without transformers
Regimes (see nn.store.REGIMES): +x+ and -x- give positive h, +x- and -x+ negative
h; the first sign is the gate's, the second the up projection's. Values are stored
as magnitudes within their pole. DESIGN_DOC.md has the size arithmetic.
"""

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .model import decoder_layers, hooked, load, separator
from .store import parse_neuron
from .textsource import FINE_WEB, Texts, describe, normalize


def _targets(neurons, L, width):
    """Sorted, de-duplicated [(layer, idx)] for a targeted harvest, or None for a full one."""
    if neurons is None:
        return None
    if isinstance(neurons, str):
        raise ValueError("neurons must be a list of ids like L12.N3456, not a single string")
    targets = []
    for s in neurons:
        l, i = parse_neuron(s)
        if not (0 <= l < L and 0 <= i < width):
            raise ValueError(f"neuron {s} out of range: {L} layers x {width} neurons")
        targets.append((l, i))
    if not targets:
        raise ValueError("neurons is an empty list; pass None to harvest every neuron")
    if len(set(targets)) != len(targets):
        raise ValueError("neurons lists the same neuron twice")
    return sorted(targets)


def harvest(store, model_name, tokens, ctx=4096, batch=8, k=20, window=32, before=24,
            fire=0.1, checkpoint=50_000_000, shard=(0, 1), device="cuda", neurons=None,
            source=None, max_docs=0):
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    model, tok = load(model_name, device)
    width = decoder_layers(model)[0].mlp.down_proj.in_features
    L = len(decoder_layers(model))
    targets = _targets(neurons, L, width)
    source = normalize(source) if source is not None else dict(FINE_WEB)
    meta = {"model": model_name, "ctx": ctx, "k": k, "window": window, "before": before, "fire": fire,
            "shard": list(shard), "layers": L, "width": width, "tokens": 0, "docs": 0,
            "source": source, "max_docs": max_docs, "source_state": None, "buffer": []}
    if targets is not None:
        meta["neurons"] = [[l, i] for l, i in targets]
    if (store / "meta.json").exists():
        old = json.loads((store / "meta.json").read_text())
        fixed = ("model", "ctx", "k", "window", "before", "fire", "shard")
        if any(old[f] != meta[f] for f in fixed):
            raise SystemExit(f"{store} holds a harvest with other settings: { {f: old[f] for f in fixed} }")
        if old.get("source", dict(FINE_WEB)) != source:
            raise SystemExit(f"{store} was harvested from {describe(old.get('source', dict(FINE_WEB)))}; "
                             f"use the same --dataset/--texts options or a new directory")
        old_targets = old.get("neurons")
        if targets is None and old_targets is not None:
            raise SystemExit(f"{store} is a targeted store of {len(old_targets)} neurons; "
                             f"resume it with the same neurons= list, or harvest a full store elsewhere")
        if targets is not None and old_targets is None:
            raise SystemExit(f"{store} is a full store; a targeted harvest needs a new directory")
        if targets is not None and [list(t) for t in old_targets] != [[l, i] for l, i in targets]:
            raise SystemExit(f"{store} is a targeted store of {len(old_targets)} other neurons; "
                             f"resume it with the same neurons= list, or use a new directory")
        meta = old
        meta["source"] = source
        vals, wins, fires = (torch.from_numpy(np.load(store / f)).to(device)
                             for f in ("vals.npy", "windows.npy", "fires.npy"))
        print(f"resuming at {meta['tokens']:,} tokens", flush=True)
    else:
        if targets is None:
            vals = torch.full((L, width, 4, k), -torch.inf, device=device)
            wins = torch.full((L, width, 4, k, window), -1, dtype=torch.int32, device=device)
        else:
            vals = torch.full((len(targets), 4, k), -torch.inf, device=device)
            wins = torch.full((len(targets), 4, k, window), -1, dtype=torch.int32, device=device)
        fires = torch.zeros((L, width, 2), dtype=torch.int64, device=device)
        tok.backend_tokenizer.save(str(store / "tokenizer.json"))

    src = Texts(source, shard, max_docs)
    state = meta.get("source_state")
    if state is None and meta.get("ds_state"):  # stores harvested before the source option existed
        state = {"hf": meta["ds_state"]}
    src.restore(state)
    buf = meta["buffer"]
    offsets = torch.arange(window, device=device) - before + window  # into xp, padded by `window`
    rows_by_layer = None
    if targets is not None:  # subset rows and model channels to index h/gate with, per layer
        rows_by_layer = {}
        for row, (l, i) in enumerate(targets):
            rows_by_layer.setdefault(l, ([], []))
            rows_by_layer[l][0].append(row)
            rows_by_layer[l][1].append(i)
        rows_by_layer = {l: (torch.tensor(rows, device=device), torch.tensor(chans, device=device))
                         for l, (rows, chans) in rows_by_layer.items()}

    xp = None  # padded token ids of the current forward pass; set by step() and read by merge()

    def merge(l, h, gate):
        h, gpos = h[:, 1:], gate[:, 1:] > 0
        if rows_by_layer is None:
            fires[l, :, 0] += (h > fire).sum((0, 1))
            fires[l, :, 1] += (h < -fire).sum((0, 1))
            dst, src_v = l, vals[l]
        else:
            if l not in rows_by_layer:
                return  # this layer has no tracked neuron, so skip all of its work
            dst, chans = rows_by_layer[l]
            h, gpos = h[..., chans], gpos[..., chans]  # only the tracked channels
            fires[l, :, 0].index_add_(0, chans, (h > fire).sum((0, 1)))
            fires[l, :, 1].index_add_(0, chans, (h < -fire).sum((0, 1)))
            src_v = vals[dst]
        cand_v, cand_p = [], []
        for s in (h, -h):
            for g in (True, False):  # regime order matches REGIMES: gate + then gate -
                v, p = s.masked_fill(gpos != g, -torch.inf).max(1)  # [B, C]
                cand_v.append(v)
                cand_p.append(p + 1)
        cv = torch.stack(cand_v, 1).permute(2, 1, 0).float()  # [C, 4, B]
        cp = torch.stack(cand_p, 1).permute(2, 1, 0)
        b = torch.arange(cp.shape[-1], device=device)[None, None, :, None]
        cw = xp[b, cp[..., None] + offsets]  # [C, 4, B, W]
        top = torch.cat([src_v, cv], -1).topk(k, -1)
        if rows_by_layer is None:
            vals[l] = top.values
            wins[l] = torch.cat([wins[l], cw], 2).gather(2, top.indices[..., None].expand(-1, -1, -1, window))
        else:
            vals[dst] = top.values
            wins[dst] = torch.cat([wins[dst], cw], 2).gather(
                2, top.indices[..., None].expand(-1, -1, -1, window))

    def save():
        meta["source_state"], meta["buffer"] = src.state_dict(), buf
        meta["max_docs"] = max_docs
        meta["docs"] = src.docs
        files = {"vals.npy": vals, "windows.npy": wins, "fires.npy": fires}
        for name, t in files.items():  # write everything first so a crash leaves the old checkpoint whole
            np.save(store / f"tmp.{name}", t.cpu().numpy())
        (store / "tmp.meta.json").write_text(json.dumps(meta))
        for name in [*files, "meta.json"]:
            os.replace(store / f"tmp.{name}", store / name)
        print(f"saved {meta['tokens']:,} tokens from {src.docs:,} documents", flush=True)

    def step(seqs):
        nonlocal xp
        x = torch.tensor(seqs, device=device)
        xp = torch.nn.functional.pad(x.int(), (window, window), value=-1)
        with torch.inference_mode(), hooked(model, merge):
            model(x, use_cache=False, logits_to_keep=1)
        meta["tokens"] += x.numel()
        return x.numel()

    sep, docs = separator(tok), iter(src)
    last_save, t0, seen0 = meta["tokens"], time.time(), meta["tokens"]
    exhausted = False
    while meta["tokens"] < tokens:
        seqs = []
        while len(seqs) < batch:
            while len(buf) < ctx:
                doc = next(docs, None)
                if doc is None:
                    break
                buf.extend([sep] + tok(doc, add_special_tokens=False).input_ids)
            if len(buf) < ctx:
                break
            seqs.append(buf[:ctx])
            del buf[:ctx]
        if not seqs:
            print(f"{describe(src.spec)} exhausted", flush=True)
            exhausted = True
            break
        step(seqs)
        if meta["tokens"] - last_save >= checkpoint:
            save()
            last_save = meta["tokens"]
            rate = (meta["tokens"] - seen0) / (time.time() - t0)
            print(f"{rate:,.0f} tokens/s", flush=True)
    if exhausted and buf:  # a finite corpus: score the partial tail as one final sequence
        step([buf])
        buf = []
    if meta["tokens"] == 0:
        print("no tokens scored: the text source was empty", flush=True)
    if meta["tokens"] > last_save or not (store / "meta.json").exists():
        save()


def merge_stores(out, inputs):
    """Combine stores harvested on disjoint shards into one (top-k of the union, summed counts)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metas = [json.loads((Path(p) / "meta.json").read_text()) for p in inputs]
    targeted = [str(p) for p, m in zip(inputs, metas) if "neurons" in m]
    if targeted:
        raise SystemExit("merging targeted stores is not supported: " + ", ".join(targeted))
    fixed = ("model", "ctx", "k", "window", "before", "fire")
    if any(m[f] != metas[0][f] for m in metas for f in fixed):
        raise SystemExit("stores were harvested with different settings")
    k = metas[0]["k"]
    arrays = {f: [np.load(Path(p) / f, mmap_mode="r") for p in inputs] for f in ("vals.npy", "windows.npy", "fires.npy")}
    vals = np.empty_like(arrays["vals.npy"][0])
    wins = np.empty_like(arrays["windows.npy"][0])
    for l in range(vals.shape[0]):  # one layer at a time to bound memory
        v = np.concatenate([a[l] for a in arrays["vals.npy"]], -1)
        w = np.concatenate([a[l] for a in arrays["windows.npy"]], -2)
        order = np.argsort(-v, -1, kind="stable")[..., :k]
        vals[l] = np.take_along_axis(v, order, -1)
        wins[l] = np.take_along_axis(w, order[..., None], -2)
    np.save(out / "vals.npy", vals)
    np.save(out / "windows.npy", wins)
    np.save(out / "fires.npy", sum(np.asarray(a) for a in arrays["fires.npy"]))
    meta = dict(metas[0], tokens=sum(m["tokens"] for m in metas), shard=None, ds_state=None, buffer=[],
                merged_from=[str(p) for p in inputs])
    (out / "meta.json").write_text(json.dumps(meta))
    shutil.copy(Path(inputs[0]) / "tokenizer.json", out / "tokenizer.json")
