"""Model loading, the MLP hook, and the per-neuron experiments the server runs.

A neuron is one channel of h = act_fn(gate_proj(x)) * up_proj(x), the input to a
decoder layer's down_proj. Every decoder MLP is patched so a callback sees (layer,
h, gate) and may return an edited h; harvesting, reading, steering, ablating and
attribution are all callbacks.

Every text starts with the document separator, as each document does during
harvesting, and position 0 (that separator) never counts as a peak.
"""

import contextlib
import math
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class _Stop(Exception):
    """Raised by a callback to end a forward pass once the target layer is done."""


def load(name, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval().requires_grad_(False)
    for i, layer in enumerate(decoder_layers(model)):
        mlp = layer.mlp
        mlp.nn_layer, mlp.nn_fn = i, None
        mlp.forward = types.MethodType(_mlp_forward, mlp)
    return model, tok


def decoder_layers(model):
    """The longest ModuleList of blocks whose .mlp is gated (gate_proj, up_proj, down_proj)."""
    lists = [m for m in model.modules() if isinstance(m, torch.nn.ModuleList) and len(m)
             and all(hasattr(getattr(m[0], "mlp", None), p) for p in ("gate_proj", "up_proj", "down_proj"))]
    return max(lists, key=len)


def attn_outputs(layer):
    """(name, output projection, head size) of a block's attention: o_proj or out_proj."""
    for name, mod in layer.named_children():
        proj = getattr(mod, "o_proj", None) or getattr(mod, "out_proj", None)
        if proj is not None:
            return name, proj, getattr(mod, "head_dim", None) or mod.head_v_dim
    raise ValueError(f"no attention output projection in {type(layer).__name__}")


def _mlp_forward(self, x):
    gate = self.gate_proj(x)
    h = self.act_fn(gate) * self.up_proj(x)
    if self.nn_fn is not None:
        out = self.nn_fn(self.nn_layer, h, gate)
        h = h if out is None else out
    return self.down_proj(h)


@contextlib.contextmanager
def hooked(model, fn):
    mlps = [layer.mlp for layer in decoder_layers(model)]
    for mlp in mlps:
        mlp.nn_fn = fn
    try:
        yield
    except _Stop:
        pass
    finally:
        for mlp in mlps:
            mlp.nn_fn = None


def separator(tok):
    """Document separator: <|endoftext|> when the vocabulary has it, else EOS."""
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    return sep if isinstance(sep, int) and sep != tok.unk_token_id else tok.eos_token_id


def encode(tok, text):
    return [separator(tok)] + tok(text, add_special_tokens=False).input_ids


def _input(model, ids):
    return torch.tensor([ids], device=model.device)


@torch.inference_mode()
def run(model, tok, layer, idx, texts=(), ids=()):
    """Per-token activation and gate pre-activation of one neuron for each text, then each token-id list.

    A token-id list is cut after its last separator, so it runs as a document of its own.
    """
    sep, out = separator(tok), []
    cut = [i[max((j + 1 for j, t in enumerate(i) if t == sep), default=0):] for i in ids]
    for seq in [encode(tok, t) for t in texts] + [[sep] + i for i in cut]:
        rec = {}

        def fn(l, h, gate):
            if l == layer:
                rec["acts"], rec["gate"] = h[0, :, idx].float().tolist(), gate[0, :, idx].float().tolist()
                raise _Stop

        with hooked(model, fn):
            model(_input(model, seq), use_cache=False, logits_to_keep=1)
        out.append({"tokens": [tok.decode([i]) for i in seq], **rec})
    return out


@torch.inference_mode()
def effect(model, tok, layer, idx, text, scale, top=8):
    """Next-token changes when the neuron's activation is multiplied by `scale` at every position.

    Reports, at the last position, the tokens whose probability and logit move most,
    and the positions where the log-probability of the actual next token moves most.
    """
    ids = encode(tok, text)
    x = _input(model, ids)
    rec = {}

    def fn(l, h, gate):
        if l == layer:
            rec["act"] = h[0, -1, idx].item()
            h[..., idx] *= scale

    base_logits = model(x, use_cache=False).logits[0].float()
    with hooked(model, fn):
        new_logits = model(x, use_cache=False).logits[0].float()
    base, new = base_logits.log_softmax(-1), new_logits.log_softmax(-1)
    dp = new[-1].exp() - base[-1].exp()
    dlogit = new_logits[-1] - base_logits[-1]
    dnext = (new[:-1] - base[:-1]).gather(1, x[0, 1:, None])[:, 0]  # log p(actual next token)
    worst = dnext.abs().topk(min(5, len(dnext)))

    def rows(order):
        i = order.indices.tolist()
        return [{"token": tok.decode([t]), "p_base": pb, "p_new": pn, "dlogit": dl} for t, pb, pn, dl in zip(
            i, base[-1, i].exp().tolist(), new[-1, i].exp().tolist(), dlogit[i].tolist())]

    return {
        "act": rec["act"], "scale": scale, "last_token": tok.decode([ids[-1]]),
        "base_top": rows(base[-1].topk(top)), "new_top": rows(new[-1].topk(top)),
        "prob_up": rows(dp.topk(top)), "prob_down": rows((-dp).topk(top)),
        "logit_up": rows(dlogit.topk(top)), "logit_down": rows((-dlogit).topk(top)),
        "next_token": [{"pos": p + 1, "token": tok.decode([ids[p + 1]]), "dlogp": dnext[p].item()}
                       for p in worst.indices.tolist()],
    }


_MODES = ("add", "set", "scale")
_REFS = ("sigma", "absmax", "resid")
_POSITIONS = ("all", "new", "prompt")
_CAL_EPS = 1e-6


def _steer_plan(interventions):
    """Normalize one intervention or an iterable of them to [(mode, strength), ...]."""
    if interventions is None:
        return []
    if isinstance(interventions, dict):
        interventions = [interventions]
    plan = []
    for n, item in enumerate(interventions):
        if not isinstance(item, dict):
            raise ValueError(f"intervention {n} is not a mapping: {item!r}")
        mode, strength = item.get("mode"), item.get("strength")
        if mode not in _MODES:
            raise ValueError(f"intervention {n}: mode must be one of {', '.join(_MODES)}, got {mode!r}")
        if isinstance(strength, bool):
            raise ValueError(f"intervention {n}: strength must be a finite number, got {strength!r}")
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            raise ValueError(f"intervention {n}: strength must be a finite number, got {strength!r}")
        if not math.isfinite(strength):
            raise ValueError(f"intervention {n}: strength must be finite, got {strength!r}")
        plan.append((mode, strength))
    return plan


@torch.inference_mode()
def generate(model, tok, layer, idx, text, interventions=(), positions="all", ref="sigma",
             max_new_tokens=80, temperature=0.0, seed=0):
    """Generate a baseline completion and one completion per calibrated intervention.

    Calibration is one clean forward pass that measures the neuron on the prompt's
    content positions and returns `mean`, `std`, `absmax`, `unit`, `content_positions`
    and `ref_unit`. `unit` is the population std (or 0.1 * absmax when that is 0).
    A strength unit is `unit` for ref sigma, the prompt's absmax for absmax, and for
    resid the layer's mean residual norm divided by the neuron's write-column norm.
    `ref_unit` is that raw factor; resid also reports `resid_norm` and `w_norm`.

    add does h[..., idx] += strength * ref_unit, set does h[..., idx] =
    strength * ref_unit, scale does h[..., idx] *= strength. A single mapping is a
    one-element sequence; runs are independent and never composed. `positions` is
    'all' (prefill and decode), 'new' (decode only; the first generated token still
    comes from the clean prefill) or 'prompt' (prefill only). BASE and every run
    share a seed, so at temperature 0 only the intervention differs.
    """
    layers = decoder_layers(model)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < len(layers):
        raise ValueError(f"layer out of range: {layer!r} (0..{len(layers) - 1})")
    width = layers[0].mlp.down_proj.in_features
    if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < width:
        raise ValueError(f"neuron out of range: {idx!r} (0..{width - 1})")
    if ref not in _REFS:
        raise ValueError(f"ref must be one of {', '.join(_REFS)}, got {ref!r}")
    if positions not in _POSITIONS:
        raise ValueError(f"positions must be one of {', '.join(_POSITIONS)}, got {positions!r}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be an integer >= 1, got {max_new_tokens!r}")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError(f"temperature must be a number >= 0, got {temperature!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
    plan = _steer_plan(interventions)

    ids = encode(tok, text)
    x = _input(model, ids)
    acts = []
    resid_norms = []

    def record(l, h, gate):
        if l == layer:
            acts.append(h[0, 1:, idx].float().reshape(-1).cpu())
            raise _Stop

    def layer_input(module, args, kwargs):
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is not None and hidden.shape[1] > 1:
            resid_norms.append(hidden[0, 1:].float().norm(dim=-1).mean().item())

    pre = layers[layer].register_forward_pre_hook(layer_input, with_kwargs=True) if ref == "resid" else None
    try:
        with hooked(model, record):
            model(x, use_cache=False)
    finally:
        if pre is not None:
            pre.remove()
    if not acts:
        raise ValueError("empty prompt: no content positions to calibrate on")
    a = torch.cat(acts)
    if not a.numel():
        raise ValueError("empty prompt: no content positions to calibrate on")
    mean = float(a.mean().item())
    std = float(a.std(unbiased=False).item())
    absmax = float(a.abs().max().item())
    unit = float(std if std > _CAL_EPS else absmax * 0.1)
    calibration = {"mean": mean, "std": std, "absmax": absmax, "unit": unit,
                   "content_positions": int(a.numel()), "ref": ref}
    if ref == "resid":
        if not resid_norms:
            raise ValueError("could not measure the residual stream for ref=resid")
        resid_norm = float(resid_norms[0])
        w_norm = float(layers[layer].mlp.down_proj.weight[:, idx].detach().float().norm().item())
        if not math.isfinite(resid_norm) or resid_norm <= 0 or not math.isfinite(w_norm) or w_norm <= 0:
            raise ValueError("residual norm and write-column norm must be positive for ref=resid")
        ref_unit = resid_norm / w_norm
        calibration.update({"resid_norm": resid_norm, "w_norm": w_norm, "ref_unit": ref_unit})
    else:
        ref_unit = float(absmax if ref == "absmax" else unit)
        calibration["ref_unit"] = ref_unit

    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, use_cache=True)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kwargs.update(do_sample=False)

    torch.manual_seed(seed)
    base = model.generate(x, **kwargs)
    base_text = tok.decode(base[0, len(ids):], skip_special_tokens=True)

    def hook(mode, strength):
        def fn(l, h, gate):
            if l != layer:
                return
            prefill = h.shape[1] > 1
            if (positions == "prompt" and not prefill) or (positions == "new" and prefill):
                return
            if mode == "add":
                h[..., idx] = h[..., idx] + strength * ref_unit
            elif mode == "set":
                h[..., idx] = strength * ref_unit
            else:
                h[..., idx] = h[..., idx] * strength
        return fn

    runs = []
    for mode, strength in plan:
        torch.manual_seed(seed)
        with hooked(model, hook(mode, strength)):
            out = model.generate(x, **kwargs)
        activation = (mean + strength * ref_unit if mode == "add"
                      else strength * ref_unit if mode == "set" else mean * strength)
        runs.append({"mode": mode, "strength": strength, "ref": ref,
                     "activation": activation,
                     "steered": tok.decode(out[0, len(ids):], skip_special_tokens=True)})
    return {"text": text, "positions": positions, "ref": ref, "calibration": calibration,
            "base": base_text, "runs": runs}


def attrib(model, tok, layer, idx, pole, text=None, ids=None, pos=None, top=15):
    """Upstream neurons and attention heads ranked by gradient x activation.

    Give `text`, or token `ids` (the separator is prepended to either).

    The target is the neuron's value in `pole` (sign-flipped for "-") at `pos`
    (default: its peak). An upstream neuron's score is sum over positions of
    d(target)/dh * h; a head's is the same over its slice of the attention output
    projection's input. Position 0 (the separator, an attention sink with huge
    activations that is the same in every text) is left out of the sums. This is a
    first-order estimate of each unit's total effect through all paths, not a
    direct-connection weight, and it can be off where the model is far from linear.
    """
    ids = encode(tok, text) if ids is None else [separator(tok)] + ids
    sign = 1.0 if pole == "+" else -1.0
    layers = decoder_layers(model)
    acts, heads, rec = {}, {}, {}

    def fn(l, h, gate):
        if l < layer:
            h.retain_grad()
            acts[l] = h
        else:
            rec["t"] = h[0, :, idx] * sign
            raise _Stop

    def save_input(l, name, size):
        def hook(mod, args):
            args[0].retain_grad()
            heads[l] = (name, size, args[0])
        return hook

    handles = [model.get_input_embeddings().register_forward_hook(lambda m, a, out: out.requires_grad_())]
    for l in range(layer + 1):
        name, proj, size = attn_outputs(layers[l])
        handles.append(proj.register_forward_pre_hook(save_input(l, name, size)))
    try:
        with torch.enable_grad(), hooked(model, fn):
            model(_input(model, ids), use_cache=False, logits_to_keep=1)
        t = rec["t"]
        pos = int(t[1:].argmax()) + 1 if pos is None else pos
        t[pos].backward()
    finally:
        for hd in handles:
            hd.remove()

    def best(scores, label):
        """scores: [T, units] per-position contributions; top units by |sum over positions 1..T-1|."""
        scores = scores[1:]
        total = scores.sum(0)
        order = total.abs().topk(min(top, total.numel())).indices
        at = scores[:, order].abs().argmax(0) + 1
        return [{"unit": label(int(u)), "attr": total[u].item(), "pos": int(p), "token": tok.decode([ids[p]])}
                for u, p in zip(order.tolist(), at.tolist())]

    width = layers[0].mlp.down_proj.in_features
    neurons = []
    if acts:
        per_pos = torch.cat([(a.grad * a)[0].float() for a in acts.values()], 1)
        neurons = best(per_pos, lambda u: f"L{u // width}.N{u % width}")
        for r in neurons:
            l, n = (int(s[1:]) for s in r["unit"].split("."))
            r["act"] = acts[l][0, r["pos"], n].item()
    hs = [((x.grad * x)[0].float().unflatten(-1, (-1, size)).sum(-1), l, name) for l, (name, size, x) in heads.items()]
    labels = [(l, h, name) for g, l, name in hs for h in range(g.shape[1])]
    head_rows = best(torch.cat([g for g, _, _ in hs], 1), lambda u: "L{}.H{} ({})".format(*labels[u]))
    return {"pos": pos, "token": tok.decode([ids[pos]]), "act": t[pos].item() * sign,
            "neurons": neurons, "heads": head_rows}
