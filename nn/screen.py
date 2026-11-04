"""Rank neurons by how specifically an intent corpus activates them over a benign one.

A screen streams two text sources (the intent/positive corpus and a benign/negative
corpus), keeps equal-length token prefixes, runs the model once over all prefixes,
and accumulates each neuron's per-text maximum over content positions. Position 0
(the separator) and position 1 (the first content token) are skipped, so the
ranking is about what the text says rather than how it starts. Neurons are ranked
by a Welch t-statistic between the two classes, and the token at each neuron's
strongest positive and negative peak is kept as evidence.

The output is a JSON document, not a store: `nn harvest --neurons` consumes the
`neurons` list, and a workflow can inspect the peak tokens before choosing which
neurons to carry into the blind test.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .model import decoder_layers, hooked, load, separator
from .textsource import Texts, describe


def _collect(tok, spec, max_texts, tokens, match=None, max_docs=0):
    """Up to max_texts token-id prefixes of exactly `tokens` content tokens."""
    import re

    pat = re.compile(match) if match else None
    out = []
    for text in Texts(spec, max_docs=max_docs):
        if len(out) >= max_texts:
            break
        if pat and not pat.search(text):
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) < tokens:
            continue
        out.append(ids[:tokens])
    return out


def screen(model_name, positive, negative, intent="", max_texts=120, tokens=100, top_k=60,
           batch=16, device="cuda", match=None, benign_match=None, min_mean=0.5,
           max_docs=0, benign_max_docs=0):
    """Run the intent-vs-benign screen and return the ranked result as a dict."""
    if tokens < 3:
        raise ValueError("tokens must be at least 3 (one separator and two content positions are skipped)")
    if max_texts < 2:
        raise ValueError("max_texts must be at least 2")
    model, tok = load(model_name, device)
    pos = _collect(tok, positive, max_texts, tokens, match, max_docs)
    neg = _collect(tok, negative, max_texts, tokens, benign_match, benign_max_docs)
    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(f"need at least 2 texts per class, got {len(pos)} positive and {len(neg)} negative "
                         f"(try a larger --max-texts or a lower --tokens)")

    sep = separator(tok)
    layers = decoder_layers(model)
    L, I = len(layers), layers[0].mlp.down_proj.in_features
    np_, nn_ = len(pos), len(neg)
    N = np_ + nn_
    vals = torch.zeros(L, I, N, device=device)
    peak_pos = torch.zeros(L, I, device=device)
    peak_neg = torch.zeros(L, I, device=device)
    tok_pos = torch.zeros(L, I, dtype=torch.long, device=device)
    tok_neg = torch.zeros(L, I, dtype=torch.long, device=device)
    holder = {"i": 0, "ids": None}

    def fn(l, h, gate):
        i = holder["i"]
        if h.shape[1] <= 2:
            return
        v, j = h[:, 2:].max(dim=1)  # batch x width, argmax within positions 2..
        v = v.float()
        vals[l, :, i:i + v.shape[0]] = v.t()
        ids = holder["ids"]
        for b in range(v.shape[0]):
            if i + b < np_:
                mask = v[b] > peak_pos[l]
                if mask.any():
                    peak_pos[l][mask] = v[b][mask]
                    tok_pos[l][mask] = ids[b][j[b][mask] + 2]
            else:
                mask = v[b] > peak_neg[l]
                if mask.any():
                    peak_neg[l][mask] = v[b][mask]
                    tok_neg[l][mask] = ids[b][j[b][mask] + 2]

    seqs = [[sep] + ids for ids in pos + neg]
    for start in range(0, N, batch):
        chunk = seqs[start:start + batch]
        holder["i"] = start
        holder["ids"] = torch.tensor(chunk, device=device)
        with torch.inference_mode(), hooked(model, fn):
            model(holder["ids"], use_cache=False, logits_to_keep=1)

    vc = vals.reshape(L * I, N)
    mh, ms = vc[:, :np_].mean(1), vc[:, np_:].mean(1)
    vh, vs = vc[:, :np_].var(1, unbiased=True), vc[:, np_:].var(1, unbiased=True)
    t = (mh - ms) / torch.sqrt(vh / np_ + vs / nn_ + 1e-9)
    mxh, mxs = vc[:, :np_].max(1).values, vc[:, np_:].max(1).values
    pv, pt = peak_pos.reshape(-1).cpu(), tok_pos.reshape(-1)
    nv, nt = peak_neg.reshape(-1).cpu(), tok_neg.reshape(-1)
    mh, ms, mxh, mxs, t = (x.cpu() for x in (mh, ms, mxh, mxs, t))

    keep = np.isfinite(t.numpy()) & (mh.numpy() >= min_mean) & (mxh.numpy() >= 1.0)
    order = np.argsort(-t.numpy(), kind="stable")
    rows = []
    for r in order:
        if not keep[r]:
            continue
        l, i = divmod(int(r), I)
        rows.append({
            "neuron": f"L{l}.N{i}",
            "t": round(float(t[r]), 3),
            "mean_pos": round(float(mh[r]), 3),
            "mean_neg": round(float(ms[r]), 3),
            "max_pos": round(float(mxh[r]), 2),
            "max_neg": round(float(mxs[r]), 2),
            "peak_pos": tok.decode([int(pt[r])]) if int(pt[r]) >= 0 else "",
            "peak_neg": tok.decode([int(nt[r])]) if int(nt[r]) >= 0 else "",
        })
        if len(rows) >= top_k:
            break

    return {
        "model": model_name,
        "intent": intent,
        "positive": describe(positive),
        "negative": describe(negative),
        "texts": {"positive": np_, "negative": nn_, "tokens": tokens},
        "neurons": rows,
    }


def write(path, result):
    path = Path(path)
    path.write_text(json.dumps(result, indent=1))
    return path
