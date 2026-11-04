"""The shareable steering showcase: one judged generation per documented neuron.

`nn steering save` validates one record before anything is written; the markdown
report's `## Showcase` section and the library page render it. A dose always shows
its unit (sigma, absmax or resid), and the judge's classification is copied
unchanged, so an incoherent run stays labelled as evidence, not a recommended dose.
"""

import re
import time
from math import isfinite

from .store import parse_neuron

STEER_MODES = ("add", "set", "scale")
STEER_POSITIONS = ("all", "new", "prompt")
STEER_REFS = ("sigma", "absmax", "resid")
STEER_LIMITS = {"query": 2000, "base": 8000, "steered": 8000,
                "selection": 300, "recipe": 1500, "reproduce": 4000, "note": 500}
STEER_RUN = re.compile(r"[A-Za-z0-9_.-]+")
STEER_FIELDS = ("neuron", "run", "model", "query", "mode", "position", "strength", "unit",
                "activation", "temperature", "seed", "chat", "base", "steered", "control",
                "classification", "selection", "recipe", "reproduce", "time", "ref", "ref_unit")
STEER_INPUT = tuple(f for f in STEER_FIELDS if f not in ("run", "model", "time", "ref", "ref_unit"))
STEER_CONTROL = ("mode", "position", "strength", "text", "ref", "ref_unit")
STEER_CLASS = ("coherent", "on_concept", "unaligned", "movement", "note")


def md_inline(s):
    """Arbitrary text as one line of Markdown prose: newlines folded, Markdown characters escaped."""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    s = " ".join(s.split("\n"))
    for ch in "\\`*_~[]()<>&|\"#$":
        s = s.replace(ch, "\\" + ch)
    return s


def period(s):
    """s with a closing period, unless it already ends in . ! or ?."""
    s = (s or "").rstrip()
    return s if not s or s[-1] in ".!?" else s + "."


def md_quote(s):
    """Blockquote the lines of s, escaped, for the full BASE and steered generations."""
    lines = (s or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").split("\n")
    return "\n".join("> " + md_inline(line) if line else ">" for line in lines)


def md_fence(s, lang="bash"):
    """A fence around s that its own backtick runs cannot close."""
    longest = max((len(m) for m in re.findall(r"`+", s or "")), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{(s or '').rstrip()}\n{fence}"


def dose_md(r, detail=False):
    """A dose as prose: 'add +128σ at positions all', or 'scale ×2 at positions new'."""
    strength = r["strength"]
    ref = r.get("ref", "sigma")
    unit = "σ" if ref == "sigma" else f" {ref}"
    out = (f"scale ×{strength:g}" if r["mode"] == "scale"
           else f"{r['mode']} {strength:+g}{unit}") + f" at positions {r['position']}"
    if detail and "activation" in r:
        if ref != "sigma" and r.get("ref_unit") is not None:
            out += (f" (1 {ref} = {r['ref_unit']:.4f} activation, "
                    f"≈ {r['activation']:+.2f} activation)")
        elif "unit" in r:
            out += f" (unit {r['unit']:.4f}, ≈ {r['activation']:+.2f} activation)"
    return out


def judge_md(r):
    """The judge's classification of a steering record, in full and without softening it."""
    c = r["classification"]
    out = (f"Judge: coherent **{'yes' if c['coherent'] else 'no'}**, "
           f"on-concept **{'yes' if c['on_concept'] else 'no'}**, "
           f"unaligned **{'yes' if c['unaligned'] else 'no'}**, movement **{c['movement']}/3**.")
    if c["note"]:
        out += " " + period(md_inline(c["note"]))
    if not c["coherent"]:
        out += " The judge called this run incoherent; it is shown for its concept shift, not as a usable answer."
    if c["unaligned"]:
        out += " The judge marked it unaligned."
    return out


def showcase_md(shows, run=None):
    """The `## Showcase` section: one ESCAPED, shareable generation per neuron, with recipe and command."""
    out = ["", "## Showcase", "",
           "One generation per documented neuron from the run's calibrated dose probe: the full BASE, the "
           "steered completion the judge classified, and its control when one was recorded. The judge's "
           "classification is copied unchanged, so a run it called incoherent stays labelled that way: that "
           "is evidence of the concept shift, not a recommended dose. Each entry carries its recipe and the "
           "command that reproduces it." + (f" Run {run}." if run else ""), ""]
    for n in sorted(shows, key=parse_neuron):
        r = shows[n]
        meta = [f"run {r['run']}"] + ([r["model"]] if r.get("model") else []) + [
            "chat template" if r["chat"] else "raw prompt",
            f"temperature {r['temperature']:g}", f"seed {r['seed']}"]
        out += [f"### {n}", "", f"*{md_inline(' · '.join(meta))}*", "",
                f"**Query.** {md_inline(r['query'])}", "",
                "**BASE.**", "", md_quote(r["base"]), "",
                f"**STEERED, {dose_md(r, detail=True)}.**", "", md_quote(r["steered"]), ""]
        if c := r.get("control"):
            out += [f"**CONTROL, {dose_md(c)}.**", "", md_quote(c["text"]), ""]
        else:
            out += ["**CONTROL.** None recorded.", ""]
        out += [judge_md(r), "",
                f"**Why this run.** {period(md_inline(r['selection']))}", "",
                f"**Recipe.** {md_inline(r['recipe'])}", "",
                "**Reproduce.**", "", md_fence(r["reproduce"]), ""]
    return out


def steer_number(v, field):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v):
        raise ValueError(f"steering record: {field} must be a finite number")
    return float(v)


def steer_int(v, field, lo, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or not float(v).is_integer():
        raise ValueError(f"steering record: {field} must be an integer")
    i = int(v)
    if i < lo or (hi is not None and i > hi):
        raise ValueError(f"steering record: {field} must be {lo}..{hi}" if hi is not None
                         else f"steering record: {field} must be at least {lo}")
    return i


def steer_text(v, field, limit, allow_empty=False):
    if not isinstance(v, str):
        raise ValueError(f"steering record: {field} must be a string")
    s = v.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if not allow_empty and not s.strip():
        raise ValueError(f"steering record: {field} must not be empty")
    if len(s) > limit:
        raise ValueError(f"steering record: {field} is {len(s)} characters, over the {limit} limit")
    return s


def check_control(v):
    """null, or {mode, position, strength, text} plus optional ref/ref_unit: a different run as control."""
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("steering record: control must be an object or null")
    if unknown := sorted(set(v) - set(STEER_CONTROL)):
        raise ValueError(f"steering record: unknown control field(s): {', '.join(unknown)}")
    if missing := [k for k in ("mode", "position", "strength", "text") if k not in v]:
        raise ValueError(f"steering record: control needs {', '.join(missing)}")
    if v["mode"] not in STEER_MODES:
        raise ValueError(f"steering record: control.mode must be one of {', '.join(STEER_MODES)}")
    if v["position"] not in STEER_POSITIONS:
        raise ValueError(f"steering record: control.position must be one of {', '.join(STEER_POSITIONS)}")
    ref = v.get("ref", "sigma")
    if ref not in STEER_REFS:
        raise ValueError(f"steering record: control.ref must be one of {', '.join(STEER_REFS)}")
    out = {"mode": v["mode"], "position": v["position"],
           "strength": steer_number(v["strength"], "control.strength"),
           "text": steer_text(v["text"], "control.text", STEER_LIMITS["steered"]),
           "ref": ref}
    if v.get("ref_unit") is not None:
        ref_unit = steer_number(v["ref_unit"], "control.ref_unit")
        if ref_unit < 0:
            raise ValueError("steering record: control.ref_unit must be at least 0")
        out["ref_unit"] = ref_unit
    return out


def check_classification(v):
    """The judge's verdict on the steered run, exactly as it gave it."""
    if not isinstance(v, dict):
        raise ValueError("steering record: classification must be an object")
    if unknown := sorted(set(v) - set(STEER_CLASS)):
        raise ValueError(f"steering record: unknown classification field(s): {', '.join(unknown)}")
    if missing := [k for k in STEER_CLASS if k not in v]:
        raise ValueError(f"steering record: classification needs {', '.join(missing)}")
    for k in ("coherent", "on_concept", "unaligned"):
        if not isinstance(v[k], bool):
            raise ValueError(f"steering record: classification.{k} must be true or false")
    return {"coherent": v["coherent"], "on_concept": v["on_concept"], "unaligned": v["unaligned"],
            "movement": steer_int(v["movement"], "classification.movement", 0, 3),
            "note": steer_text(v["note"], "classification.note", STEER_LIMITS["note"], allow_empty=True)}


def check_record(a, st, payload):
    """Validate one showcase record and return it normalized; raises before anything is written."""
    if not isinstance(payload, dict):
        raise ValueError("steering record: expected a JSON object")
    if unknown := sorted(set(payload) - set(STEER_FIELDS)):
        raise ValueError(f"steering record: unknown field(s): {', '.join(unknown)}")
    if missing := [f for f in STEER_INPUT if f not in payload]:
        raise ValueError(f"steering record: missing field(s): {', '.join(missing)}")
    neuron = payload["neuron"]
    if not isinstance(neuron, str):
        raise ValueError("steering record: neuron must be a string like L8.N2373")
    layer, idx = parse_neuron(neuron)
    st.check(layer, idx)
    run = payload.get("run")
    if run is None:
        run = a.run
    elif not isinstance(run, str) or not STEER_RUN.fullmatch(run):
        raise ValueError("steering record: run must be letters, digits, _ . or -, e.g. r5")
    if a.run and run and run != a.run:
        raise ValueError(f"steering record: run {run!r} does not match --run {a.run!r}")
    if not run:
        raise ValueError('steering record: no run label; pass --run, set NN_RUN, or put "run" in the JSON')
    model = payload.get("model", st.meta["model"])
    if model != st.meta["model"]:
        raise ValueError(f"steering record: model {model!r} is not this store's model {st.meta['model']!r}")
    mode, position = payload["mode"], payload["position"]
    if mode not in STEER_MODES:
        raise ValueError(f"steering record: mode must be one of {', '.join(STEER_MODES)}")
    if position not in STEER_POSITIONS:
        raise ValueError(f"steering record: position must be one of {', '.join(STEER_POSITIONS)}")
    ref = payload.get("ref", "sigma")
    if ref not in STEER_REFS:
        raise ValueError(f"steering record: ref must be one of {', '.join(STEER_REFS)}")
    ref_unit = payload.get("ref_unit")
    if ref_unit is not None:
        ref_unit = steer_number(ref_unit, "ref_unit")
        if ref_unit < 0:
            raise ValueError("steering record: ref_unit must be at least 0")
    if ref != "sigma" and ref_unit is None:
        raise ValueError("steering record: ref_unit is required when ref is not sigma")
    unit = steer_number(payload["unit"], "unit")
    if unit < 0:
        raise ValueError("steering record: unit must be at least 0")
    temperature = steer_number(payload["temperature"], "temperature")
    if temperature < 0:
        raise ValueError("steering record: temperature must be at least 0")
    if not isinstance(payload["chat"], bool):
        raise ValueError("steering record: chat must be true or false")
    t = payload.get("time")
    return {
        "neuron": neuron, "run": run, "model": model,
        "query": steer_text(payload["query"], "query", STEER_LIMITS["query"]),
        "mode": mode, "position": position, "strength": steer_number(payload["strength"], "strength"),
        "ref": ref, "unit": unit, "activation": steer_number(payload["activation"], "activation"),
        "temperature": temperature, "seed": steer_int(payload["seed"], "seed", 0),
        "chat": payload["chat"],
        "base": steer_text(payload["base"], "base", STEER_LIMITS["base"]),
        "steered": steer_text(payload["steered"], "steered", STEER_LIMITS["steered"]),
        "control": check_control(payload["control"]),
        "classification": check_classification(payload["classification"]),
        "selection": steer_text(payload["selection"], "selection", STEER_LIMITS["selection"]),
        "recipe": steer_text(payload["recipe"], "recipe", STEER_LIMITS["recipe"]),
        "reproduce": steer_text(payload["reproduce"], "reproduce", STEER_LIMITS["reproduce"]),
        "time": time.time_ns() / 1e9 if t is None else steer_number(t, "time"),
    } | ({"ref_unit": ref_unit} if ref_unit is not None else {})
