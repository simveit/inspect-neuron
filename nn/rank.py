"""Rank neurons by how strongly a text corpus activates them.

nn.harvest stores the strongest examples of every neuron of a model, which is a
lot of GPU memory and disk for a large model. This pass accumulates cheap
per-neuron statistics instead, over one pass of the corpus, skipping position 0
like harvest. Four statistics are available and all of them are kept, so the
ranking can be changed without streaming again:

  fires  tokens whose activation h is beyond +fire / -fire (the default)
  max    the strongest positive magnitude seen in each pole
  mean   the mean positive part of h over the corpus
  spike  max - mean: neurons that spike on this corpus rather than being
         always-on. This is what "which neurons do these texts trigger" wants.

The corpus is FineWeb (sample-10BT) by default; pass `source=` (from
nn.textsource) to stream a Hugging Face dataset or local JSONL/plain-text file
instead, with `max_docs` to cap it. The source is stored in meta.json and must
match on resume.

Output (one directory):
  fires.npy   int64   [L, I, 2]  tokens with h > fire / h < -fire
  sums.npy    float32 [L, I, 2]  sum of max(h, 0) / max(-h, 0) over tokens
  peaks.npy   float32 [L, I, 2]  largest max(h, 0) / max(-h, 0) over tokens
  meta.json   model, layers, width, ctx, fire, k, shard, tokens, token_count,
              source, docs, buffer
  top.json    {"model", "tokens", "fire", "k", "stat", "source", "docs",
               "neurons": [ids sorted by the stat, descending],
               "counts": [[plus, minus], ...]}

Streaming, sharding, resuming and checkpointing are exactly nn.harvest's: rerun
with the same or a larger `tokens` to continue. When the token target is already
reached the stream is skipped and only top.json is rewritten, so changing k or
stat is cheap: `nn rank --out DIR --tokens 0 --stat spike --k 20`.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from .textsource import FINE_WEB, Texts, describe, normalize

STATS = ("fires", "max", "mean", "spike")
FIXED = ("model", "ctx", "fire", "shard")  # k/stat are free: re-emitting top.json is cheap


def _values(meta, stat, fires, sums, peaks):
    """The per-neuron, per-pole value used for ranking, as float64 [L, I, 2]."""
    if stat == "fires":
        return np.asarray(fires, dtype=np.float64)
    peak = np.asarray(peaks, dtype=np.float64)
    if stat == "max":
        return peak
    mean = np.asarray(sums, dtype=np.float64) / max(int(meta.get("token_count", 0)), 1)
    if stat == "mean":
        return mean
    if stat == "spike":
        return peak - mean
    raise ValueError(f"unknown stat {stat!r}; choose one of {', '.join(STATS)}")


def _number(value, stat):
    return int(round(value)) if stat == "fires" else round(float(value), 4)


def _emit(out, meta, stat, k, fires, sums, peaks):
    """Write top.json and meta.json from the stored statistics; return the top.json dict."""
    values = _values(meta, stat, fires, sums, peaks)
    total = values.sum(-1).ravel()
    k = max(0, min(int(k), total.size))
    order = np.argsort(-total, kind="stable")[:k]  # ties keep layer, then index order
    width = values.shape[1]
    ids = [(r // width, r % width) for r in order.tolist()]
    top = {"model": meta["model"], "tokens": meta["tokens"], "fire": meta["fire"], "k": k, "stat": stat,
           "source": meta.get("source", dict(FINE_WEB)), "docs": meta.get("docs", 0),
           "neurons": [f"L{l}.N{i}" for l, i in ids],
           "counts": [[_number(values[l, i, 0], stat), _number(values[l, i, 1], stat)] for l, i in ids]}
    meta["k"], meta["stat"] = k, stat
    meta["docs"] = meta.get("docs", 0)
    (out / "tmp.top.json").write_text(json.dumps(top))
    (out / "tmp.meta.json").write_text(json.dumps(meta))
    os.replace(out / "tmp.top.json", out / "top.json")
    os.replace(out / "tmp.meta.json", out / "meta.json")
    return top


def rank(out, model_name, tokens, k=20, fire=0.1, ctx=4096, batch=8, checkpoint=50_000_000,
         shard=(0, 1), device="cuda", source=None, max_docs=0, stat="fires"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = normalize(source) if source is not None else dict(FINE_WEB)
    if stat not in STATS:
        raise ValueError(f"unknown stat {stat!r}; choose one of {', '.join(STATS)}")
    meta_path = out / "meta.json"
    arrays = {name: out / f"{name}.npy" for name in ("fires", "sums", "peaks")}
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if old is not None:
        want = {"model": model_name, "ctx": ctx, "fire": fire, "shard": list(shard)}
        if any(old[f] != want[f] for f in FIXED):
            raise SystemExit(f"{out} holds a rank pass with other settings: { {f: old[f] for f in FIXED} }")
        if old.get("source", dict(FINE_WEB)) != source and old["tokens"] < tokens:
            raise SystemExit(f"{out} holds a rank pass over {describe(old.get('source', dict(FINE_WEB)))}; "
                             f"use the same --dataset/--texts options or a new directory")
        if old["tokens"] >= tokens:  # target reached: skip the stream and just rank again
            loaded = {name: np.load(path) if path.exists() else None for name, path in arrays.items()}
            zeros = np.zeros((old["layers"], old["width"], 2), dtype=np.float32)
            return _emit(out, old, stat, k,
                         loaded["fires"] if loaded["fires"] is not None else zeros,
                         loaded["sums"] if loaded["sums"] is not None else zeros,
                         loaded["peaks"] if loaded["peaks"] is not None else zeros)

    import torch

    from .model import decoder_layers, hooked, load, separator

    model, tok = load(model_name, device)
    if old is None:
        layers = decoder_layers(model)
        L, width = len(layers), layers[0].mlp.down_proj.in_features
        meta = {"model": model_name, "ctx": ctx, "fire": fire, "k": k, "stat": stat, "shard": list(shard),
                "layers": L, "width": width, "tokens": 0, "token_count": 0, "docs": 0, "source": source,
                "max_docs": max_docs, "source_state": None, "buffer": []}
        fires = torch.zeros((L, width, 2), dtype=torch.int64, device=device)
        sums = torch.zeros((L, width, 2), dtype=torch.float32, device=device)
        peaks = torch.zeros((L, width, 2), dtype=torch.float32, device=device)
    else:
        meta = old
        meta["source"] = source
        L, width = meta["layers"], meta["width"]
        fires = torch.from_numpy(np.load(arrays["fires"])).to(device)
        sums = (torch.from_numpy(np.load(arrays["sums"])).to(device) if arrays["sums"].exists()
                else torch.zeros((L, width, 2), dtype=torch.float32, device=device))
        peaks = (torch.from_numpy(np.load(arrays["peaks"])).to(device) if arrays["peaks"].exists()
                 else torch.zeros((L, width, 2), dtype=torch.float32, device=device))
        print(f"resuming at {meta['tokens']:,} tokens", flush=True)

    src = Texts(source, shard, max_docs)
    state = meta.get("source_state")
    if state is None and meta.get("ds_state"):  # rank dirs written before the source option existed
        state = {"hf": meta["ds_state"]}
    src.restore(state)
    buf = meta["buffer"]

    def merge(l, h, gate):
        h = h[:, 1:]  # position 0 is the document separator
        pos, neg = h.clamp(min=0), (-h).clamp(min=0)
        fires[l, :, 0] += (h > fire).sum((0, 1))
        fires[l, :, 1] += (h < -fire).sum((0, 1))
        sums[l, :, 0] += pos.sum((0, 1))
        sums[l, :, 1] += neg.sum((0, 1))
        peaks[l, :, 0] = torch.maximum(peaks[l, :, 0], pos.amax((0, 1)))
        peaks[l, :, 1] = torch.maximum(peaks[l, :, 1], neg.amax((0, 1)))

    def save():
        meta["source_state"], meta["buffer"] = src.state_dict(), buf
        meta["docs"], meta["max_docs"] = src.docs, max_docs
        for name, tensor in (("fires", fires), ("sums", sums), ("peaks", peaks)):
            np.save(out / f"tmp.{name}.npy", tensor.cpu().numpy())
        (out / "tmp.meta.json").write_text(json.dumps(meta))
        for name in ("fires", "sums", "peaks"):
            os.replace(out / f"tmp.{name}.npy", arrays[name])
        os.replace(out / "tmp.meta.json", meta_path)
        print(f"saved {meta['tokens']:,} tokens from {src.docs:,} documents", flush=True)

    def step(seqs):
        x = torch.tensor(seqs, device=device)
        with torch.inference_mode(), hooked(model, merge):
            model(x, use_cache=False, logits_to_keep=1)
        meta["tokens"] += x.numel()
        meta["token_count"] = meta.get("token_count", 0) + x.numel() - x.shape[0]
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
    meta["source_state"], meta["buffer"] = src.state_dict(), buf
    meta["docs"], meta["max_docs"] = src.docs, max_docs
    if meta["tokens"] > last_save:
        save()
    return _emit(out, meta, stat, k, fires.cpu().numpy(), sums.cpu().numpy(), peaks.cpu().numpy())
