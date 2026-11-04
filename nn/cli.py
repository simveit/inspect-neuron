"""nn: the tools behind the blind neuron test and its steering probe.

Agents call: top, run, steer, ablate, generate, attrib, card, paper. The workflow calls:
select, screen, score, steering save, report. Operators call: harvest, rank, merge, serve. The store
comes from --store or $NN_STORE, the server from --server or $NN_SERVER. With --run or
$NN_RUN, cards, scores, papers and steering records are stamped with that label and read
back only within it, so several runs can share one store.

  nn screen --intent violence --dataset ... --benign-dataset ... --out screen.json
  nn harvest --model Qwen/Qwen3-0.6B --neurons picks.json --tokens 2e6
  nn select 3 --seed 0; nn top L14.N1234 --pole + -n 20
  nn generate L8.N2373 --text "..." --chat --add 64,128,-64,-128 --json
  nn steering save < record.json; nn library rebuild; nn report
"""

import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from math import comb, isfinite
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # else importing numpy spins up a thread per core
import numpy as np  # noqa: E402

from .showcase import STEER_POSITIONS, STEER_REFS, check_record, showcase_md
from .store import POLE_REGIMES, REGIMES, Store, parse_neuron

REGIME_ARGS = {"++": [0], "--": [1], "+-": [2], "-+": [3], "mixed": [2, 3]}
STRONG = 0.3  # a strong response is at least 30% of the short-context max (see short_max)
ALPHA = 0.05  # level of the one-sided Fisher exact tests behind a verdict (see judge)
DEAD_EPS = 0.1  # select: a neuron is dead if neither pole's corpus max reaches this
MIN_FIRES = 20  # select: and if fewer tokens than this fired beyond the store's threshold
SETS = ("new", "near_miss", "distant")
CARD_FIELDS = ("core", "negative", "offshoots", "residue")
MAX_FIELD = 500  # characters per card field
MAX_TITLE, MAX_PARAGRAPH = 100, 1200  # characters of a paper's title and paragraph
PAPER_TOP, PAPER_PAIRS = 5, 3  # corpus examples and contrast pairs per pole in a paper


# --- helpers ---------------------------------------------------------------

def open_store(a):
    if not a.store:
        raise ValueError("no store: set NN_STORE or pass --store")
    return Store(a.store)


def neuron_arg(a, st=None):
    layer, idx = parse_neuron(a.neuron)
    if st is not None:
        st.check(layer, idx)
    return layer, idx


def post(a, path, **body):
    req = urllib.request.Request(a.server.rstrip("/") + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"nn: server error: {json.load(e).get('error')}")
    except urllib.error.URLError:
        sys.exit(f"nn: no neuron server at {a.server} (start one with `nn serve`)")


def pct(v, ref):
    return f"{100 * v / ref:.0f}%" if ref > 0 else "n/a"


def percent(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def pval(p):
    return "-" if p is None else f"{p:.3f}" if p >= 0.001 else "<0.001"


def cell(s, n=None):
    """Text for a Markdown table cell, cut to n characters."""
    s = (s or "").replace("|", "\\|").replace("\n", " ")
    return s if n is None or len(s) <= n else s[:n - 1] + "…"


def show(t):
    return repr(t.replace("\n", "⏎"))


def emit_json(obj):
    print(json.dumps(obj, indent=1))


def float_list(s):
    """Comma-separated finite numbers; an empty string is an empty list."""
    values = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            if (s or "").strip():
                raise argparse.ArgumentTypeError(f"empty value in {s!r}")
            continue
        try:
            v = float(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {part!r}")
        if not isfinite(v):
            raise argparse.ArgumentTypeError(f"not a finite number: {part!r}")
        values.append(v)
    return values


def pole_value(acts, pole):
    """Strongest activation of a text in a pole (a magnitude), skipping the separator."""
    v = np.asarray(acts[1:]) * (1 if pole == "+" else -1)
    return max(float(v.max()), 0.0) if len(v) else 0.0


def short_max(a, st, neuron, layer, idx, pole):
    """Largest pole value over the pole's stored top windows, each up to its peak and run alone after the separator.

    Blind texts are short, and the corpus max comes from deep inside long contexts,
    so this is what they are measured against. Cached in the store's short_max.json.
    """
    path = st.path / "short_max.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = f"{neuron} {pole} {st.meta['tokens']}"  # a resumed harvest has new top windows
    if key not in cache:
        b = st.meta["before"]  # the peak's index in a window
        wins = [[int(i) for i in w[:b + 1] if i >= 0] for _, _, w in st.examples(layer, idx, POLE_REGIMES[pole])]
        runs = post(a, "/run", neuron=neuron, ids=wins) if wins else []
        cache[key] = max((pole_value(r["acts"], pole) for r in runs), default=0.0)
        path.write_text(json.dumps(cache, indent=1))
    return cache[key]


def card_dir(st):
    return st.path / "cards"


def score_dir(st):
    return st.path / "scores"


def paper_dir(st):
    return st.path / "papers"


def steering_dir(st):
    return st.path / "steering"


def load_dir(a, path):
    """The JSON records in a store folder, oldest first; with --run only that run's."""
    rows = (json.loads(p.read_text()) for p in path.glob("*.json"))
    return sorted((r for r in rows if not a.run or r.get("run") == a.run), key=lambda r: r.get("time") or 0)


def newest(a, path, *keys):
    """{key tuple: newest record} over a store folder, e.g. keys ("neuron", "pole") for scores."""
    return {tuple(r[k] for k in keys): r for r in load_dir(a, path)}


# --- agent commands ----------------------------------------------------------

def cmd_top(a):
    st = open_store(a)
    layer, idx = neuron_arg(a, st)
    pole = a.pole or ("-" if a.regime in ("+-", "-+", "mixed") else "+")
    regimes = REGIME_ARGS[a.regime] if a.regime else POLE_REGIMES[pole]
    if not set(regimes) <= set(POLE_REGIMES[pole]):
        raise ValueError(f"regime {a.regime} belongs to the other pole")
    cmax = st.corpus_max(layer, idx)
    rows = st.examples(layer, idx, regimes)
    shown = rows[a.start:a.start + a.n]
    fires = st.fires[layer, idx, 0 if pole == "+" else 1]
    counts = ", ".join(f"{REGIMES[r]} {sum(row[1] == r for row in rows)}" for r in regimes)
    other = "-" if pole == "+" else "+"
    print(f"{a.neuron} pole {pole} | corpus max {cmax[pole]:.3f} (pole {other}: {cmax[other]:.3f}) | "
          f"{fires:,} tokens beyond ±{st.meta['fire']} | stored: {counts}")
    print("value, % of corpus max, regime (gate x up) | text with the peak token in [[ ]]")
    for i, (v, r, w) in enumerate(shown, a.start):
        print(f"{i:3d} {v:8.3f} {pct(v, cmax[pole]):>4} {REGIMES[r]} | {st.window_text(w)}")
    if not shown:
        print("no examples")


def cmd_run(a):
    st = open_store(a)
    layer, idx = neuron_arg(a, st)
    res = post(a, "/run", neuron=a.neuron, texts=[a.text])[0]
    cmax = st.corpus_max(layer, idx)
    toks, acts, gate = res["tokens"][1:], res["acts"][1:], res["gate"][1:]
    if not toks:
        raise ValueError("empty text")
    hi, lo = int(np.argmax(acts)), int(np.argmin(acts))
    print(f"{a.neuron} | {len(toks)} tokens | max {acts[hi]:+.3f} at {show(toks[hi])} "
          f"({pct(max(acts[hi], 0), cmax['+'])} of corpus + max {cmax['+']:.3f}) | min {acts[lo]:+.3f} at "
          f"{show(toks[lo])} ({pct(max(-acts[lo], 0), cmax['-'])} of corpus - max {cmax['-']:.3f})")
    print(f"[value] after tokens beyond {a.min:.0%} of their pole's corpus max; * = gate negative")
    out = []
    for t, v, g in zip(toks, acts, gate):
        ref = cmax["+"] if v > 0 else cmax["-"]
        mark = f"[{v:+.2f}{'*' if g <= 0 else ''}]" if ref > 0 and abs(v) >= a.min * ref else ""
        out.append(t + mark)
    print("".join(out))


def _effect(a, scale, label):
    neuron_arg(a)
    res = post(a, "/effect", neuron=a.neuron, text=a.text, scale=scale, top=a.top)

    def probs(rows, key):
        return ", ".join(f"{show(r['token'])} {100 * r[key]:.3g}%" for r in rows)

    def moves(rows):
        return ", ".join(f"{show(r['token'])} {100 * r['p_base']:.3g}->{100 * r['p_new']:.3g}%" for r in rows)

    def logits(rows):
        return ", ".join(f"{show(r['token'])} {r['dlogit']:+.2f}" for r in rows)

    print(f"{a.neuron} {label} at every position | last token {show(res['last_token'])}, "
          f"activation there {res['act']:+.3f} -> {res['act'] * scale:+.3f}")
    print(f"next token before: {probs(res['base_top'], 'p_base')}")
    print(f"next token after:  {probs(res['new_top'], 'p_new')}")
    print(f"prob up:    {moves(res['prob_up'])}")
    print(f"prob down:  {moves(res['prob_down'])}")
    print(f"logit up:   {logits(res['logit_up'])}")
    print(f"logit down: {logits(res['logit_down'])}")
    print("log p change of the text's own tokens, as position (1 = first token) and token predicted there: " + ", ".join(
        f"{r['pos']} {show(r['token'])} {r['dlogp']:+.3f}" for r in res["next_token"]))


def cmd_steer(a):
    _effect(a, a.scale, f"scaled x{a.scale:g}")


def cmd_ablate(a):
    _effect(a, 0.0, "zero-ablated")


def cmd_generate(a):
    from .model import decoder_layers, generate, load
    layer, idx = parse_neuron(a.neuron)
    model, tok = load(a.model, a.device)
    layers = decoder_layers(model)
    width = layers[0].mlp.down_proj.in_features
    if not (0 <= layer < len(layers) and 0 <= idx < width):
        raise ValueError(f"neuron out of range: {len(layers)} layers x {width} neurons")
    text = a.text
    if a.chat:
        messages = [{"role": "user", "content": text}]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:  # a tokenizer whose template does not know enable_thinking
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    interventions = [{"mode": mode, "strength": s}
                     for mode in ("add", "set", "scale") for s in getattr(a, mode)]
    res = generate(model, tok, layer, idx, text, interventions=interventions, positions=a.positions,
                   ref=a.ref, max_new_tokens=a.max_new_tokens, temperature=a.temperature, seed=a.seed)
    if a.json:
        return emit_json(res)
    cal = res["calibration"]
    print(f"{a.neuron} | {a.text.replace(chr(10), ' ')}")
    ref = cal.get("ref", "sigma")
    print(f"CALIBRATION: {a.positions} positions | {cal['content_positions']} content positions | "
          f"mean {cal['mean']:+.4f} std {cal['std']:.4f} absmax {cal['absmax']:.4f} "
          f"unit {cal['unit']:.4f}" + ("" if ref == "sigma" else f" ref {ref}"))
    if ref == "resid":
        print(f"RESIDUAL: mean norm {cal['resid_norm']:.4f} | write column {cal['w_norm']:.4f} | "
              f"one resid unit = {cal['ref_unit']:.4f} activation")
    if cal["unit"] == 0:
        print("nn: warning: unit is 0 (dead neuron on this prompt): add and scale are no-ops, set is an ablation",
              file=sys.stderr)
    print("BASE:    " + res["base"].replace("\n", " "))
    for r in res["runs"]:
        label = f"{r['mode']} {r['strength']:+g}"
        if ref != "sigma":
            label += f" {r['ref']} ({r['activation']:+.4f} activation)"
        print(f"RUN {label}: " + r["steered"].replace("\n", " "))


def cmd_attrib(a):
    st = open_store(a)
    layer, idx = neuron_arg(a, st)
    pole = a.pole
    if a.example.lstrip("-").isdigit():
        rows = st.examples(layer, idx, POLE_REGIMES[pole])
        if not 0 <= int(a.example) < len(rows):
            raise ValueError(f"example {a.example} out of range (0..{len(rows) - 1}, see `nn top`)")
        value, _, w = rows[int(a.example)]
        b = st.meta["before"]
        ids = [int(i) for i in w if i >= 0]
        pos = b - int((w[:b] < 0).sum()) + 1  # +1: the server prepends the separator
        res = post(a, "/attrib", neuron=a.neuron, pole=pole, ids=ids, pos=pos, top=a.top)
        note = f" (corpus value {value if pole == '+' else -value:+.3f} with the full context; here only the {len(ids)}-token window)"
    else:
        res = post(a, "/attrib", neuron=a.neuron, pole=pole, text=a.example, top=a.top)
        note = ""
    print(f"{a.neuron} pole {pole} at pos {res['pos']} {show(res['token'])}: activation {res['act']:+.3f}{note}")
    print("gradient x activation of each upstream unit on the target, summed over positions after the separator; "
          "positive = pushes the neuron toward this pole; 'at' = position contributing most")
    print("upstream neurons:")
    for r in res["neurons"]:
        print(f"  {r['unit']:<12} {r['attr']:+8.3f}  at {r['pos']} {show(r['token'])} (act {r['act']:+.3f})")
    print("attention heads:")
    for r in res["heads"]:
        print(f"  {r['unit']:<26} {r['attr']:+8.3f}  at {r['pos']} {show(r['token'])}")


def check_length(fields, limit=MAX_FIELD):
    if long := [f"{k} ({len(v)})" for k, v in fields.items() if len(v or "") > limit]:
        raise ValueError(f"over {limit} characters, shorten: {', '.join(long)}")


def cmd_card_submit(a):
    st = open_store(a)
    neuron_arg(a, st)
    fields = {f: getattr(a, f) for f in CARD_FIELDS}
    check_length(fields)
    card_dir(st).mkdir(exist_ok=True)
    cid = "c" + secrets.token_hex(4)
    card = {"id": cid, "neuron": a.neuron, **fields, "run": a.run, "time": time.time()}
    (card_dir(st) / f"{cid}.json").write_text(json.dumps(card, indent=1))
    print(cid)


def card_text(card):
    lines = [f"Core: {card['core']}"]
    for key, label in (("negative", "Negative pole"), ("offshoots", "Known offshoots"),
                       ("residue", "Unexplained residue")):
        if card.get(key):
            lines.append(f"{label}: {card[key]}")
    return "\n".join(lines)


def load_card(st, cid):
    path = card_dir(st) / f"{cid}.json"
    if not cid.isalnum() or not path.exists():
        raise ValueError(f"no card {cid}")
    return json.loads(path.read_text())


def blind_text(card, pole):
    """What the blind tester of one pole reads: the anonymized text from `nn card redact`, never the raw card."""
    if not (text := card.get("redacted", {}).get(pole)):
        raise ValueError(f"card {card['id']} has no anonymized {pole} pole text yet (see `nn card redact`)")
    return text


def cmd_card_read(a):
    card = load_card(open_store(a), a.card_id)
    print(blind_text(card, a.pole) if a.pole else card_text(card))


def cmd_card_redact(a):
    st = open_store(a)
    card = load_card(st, a.card_id)
    if not a.text.strip():
        raise ValueError("empty text")
    check_length({"text": a.text})
    redacted = card.setdefault("redacted", {})
    if a.pole in redacted:  # keeps what a scored tester read fixed by card id
        raise ValueError(f"card {a.card_id} already has an anonymized {a.pole} pole text; submit a new card")
    redacted[a.pole] = a.text
    (card_dir(st) / f"{a.card_id}.json").write_text(json.dumps(card, indent=1))


def cmd_steering_save(a):
    st = open_store(a)
    try:
        raw = (sys.stdin.read() if (not a.file or a.file == "-")
               else Path(a.file).read_text())
        payload = json.loads(raw)
    except (OSError, ValueError) as e:
        raise ValueError(f"steering record: cannot read JSON: {e}") from None
    rec = check_record(a, st, payload)
    steering_dir(st).mkdir(exist_ok=True)
    ns = time.time_ns()
    path = steering_dir(st) / f"{rec['neuron']}_{ns}.json"
    path.write_text(json.dumps(rec, indent=1))
    print(path)


def twins(r):
    """[(new, near miss)] score rows: each new text beside the near miss its tester wrote at the same position
    (its twin), the widest gaps first."""
    def by_tester(rows):
        groups = {}
        for x in rows:
            groups.setdefault(x.get("tester"), []).append(x)
        return groups

    new, near = by_tester(r["sets"]["new"]), by_tester(r["sets"]["near_miss"])
    return sorted(((x, y) for t in new for x, y in zip(new[t], near.get(t, []))),
                  key=lambda p: p[1]["ratio"] - p[0]["ratio"])


def contrast(r):
    """The twins with the widest gaps, for the paper writer."""
    return [{"new": x["text"], "new_act": x["act"], "new_ratio": x["ratio"],
             "near_miss": y["text"], "near_miss_act": y["act"], "near_miss_ratio": y["ratio"]}
            for x, y in twins(r)[:PAPER_PAIRS]]


def pole_scores(a, st, neuron):
    """{pole: the newest score of that pole of the neuron}."""
    return {pole: r for (n, pole), r in newest(a, score_dir(st), "neuron", "pole").items() if n == neuron}


def paper_facts(a, st, neuron):
    """What the paper writer reads: the card, and per pole the newest score, top examples and contrast.

    The card is the one the newest score tested, else the neuron's newest card.
    """
    layer, idx = parse_neuron(neuron)
    st.check(layer, idx)
    scores = pole_scores(a, st, neuron)
    tested = [r for r in scores.values() if r.get("card")]
    card = (load_card(st, max(tested, key=lambda r: r.get("time") or 0)["card"]) if tested
            else newest(a, card_dir(st), "neuron").get((neuron,)))
    cmax = st.corpus_max(layer, idx)
    poles = {}
    for pole, regimes in POLE_REGIMES.items():
        r = scores.get(pole)
        poles[pole] = {"corpus_max": cmax[pole],
                       "score": r and f"{summary_line(r)} | card {r['card'] or '-'}",
                       "top": [{"value": v, "regime": REGIMES[g], "text": st.window_text(w)}
                               for v, g, w in st.examples(layer, idx, regimes)[:PAPER_TOP]],
                       "contrast": contrast(r) if r else []}
    return {"neuron": neuron, "run": a.run, "card": card, "poles": poles}


def print_facts(f):
    run = f"run {f['run']}" if f["run"] else "all runs"
    print(f"# Facts for a paper on {f['neuron']} ({run})", "", sep="\n")
    if c := f["card"]:
        print(f"Card {c['id']}:", card_text(c), sep="\n")
        for pole, text in c.get("redacted", {}).items():
            print(f"Anonymized {pole} pole text (all its blind testers read): {text}")
    else:
        print("No card.")
    for pole, p in f["poles"].items():
        print("", f"## {pole} pole, corpus max {p['corpus_max']:.3f}", f"Score: {p['score'] or 'not scored'}", sep="\n")
        print("Top corpus examples (value, regime gate x up | text with the peak token in [[ ]]):")
        for x in p["top"]:
            print(f"  {x['value']:7.3f} {x['regime']} | {x['text']}")
        if p["contrast"]:
            print("Contrast: new texts beside the near miss written with them, widest gap first "
                  f"(value, % of the short-context max; strong is >= {STRONG:.0%}):")
            for x in p["contrast"]:
                print(f"  new  {x['new_act']:7.3f} {pct(x['new_ratio'], 1):>4} | {x['new']}")
                print(f"  near {x['near_miss_act']:7.3f} {pct(x['near_miss_ratio'], 1):>4} | {x['near_miss']}")


PAPER_TEXT = {"title": MAX_TITLE, "summary": MAX_PARAGRAPH, "clean_why": MAX_FIELD, "interesting_why": MAX_FIELD}


def cmd_paper(a):
    save = a.what == "save"
    if save != bool(a.neuron):
        raise ValueError("expected `nn paper NEURON` or `nn paper save NEURON --title ...`")
    st = open_store(a)
    facts = paper_facts(a, st, a.neuron if save else a.what)
    if not save:
        return print_facts(facts)
    if missing := [f"--{k.replace('_', '-')}" for k in (*PAPER_TEXT, "clean", "interesting") if getattr(a, k) is None]:
        raise ValueError(f"paper save needs {', '.join(missing)}")
    for k, limit in PAPER_TEXT.items():
        check_length({k: getattr(a, k)}, limit)
    paper_dir(st).mkdir(exist_ok=True)
    ns = time.time_ns()
    paper = {"neuron": facts["neuron"], "run": a.run, "card": facts["card"] and facts["card"]["id"],
             **{k: getattr(a, k) for k in PAPER_TEXT}, "clean": a.clean, "interesting": a.interesting,
             "contrast": {pole: p["contrast"] for pole, p in facts["poles"].items()}, "time": ns / 1e9}
    (paper_dir(st) / f"{facts['neuron']}_{ns}.json").write_text(json.dumps(paper, indent=1))
    show = showcase_map(a, st).get(facts["neuron"])
    print(f"saved the paper on {facts['neuron']}; {library_update(a, st, paper, show)}")


def showcase_map(a, st):
    """{neuron: newest steering record} for the run (or all runs without --run)."""
    return {n: r for (n,), r in newest(a, steering_dir(st), "neuron").items()}


def winning_runs(st, neuron):
    """Runs whose newest score on either pole of the neuron is a WIN (None is the unlabelled run).

    Reads every score record in the store, not only the current --run's, so a run-scoped library
    rebuild can tell whether another run still owns the neuron's page.
    """
    latest = {}
    for path in score_dir(st).glob("*.json"):
        r = json.loads(path.read_text())
        if r.get("neuron") != neuron:
            continue
        key = (r.get("run"), r.get("pole"))
        if key not in latest or (r.get("time") or 0) > (latest[key].get("time") or 0):
            latest[key] = r
    return {run for (run, _), r in latest.items() if r.get("verdict") == "WIN"}


def library_update(a, st, paper, show=None):
    """Write or remove the paper's neuron library page and rewrite the index; says what it did.

    The library's root is --library, else $NN_LIBRARY, else neuron-library/ beside the store.
    A run-scoped rebuild never deletes a page another run still has a winning score for;
    library.save holds the ownership rule.
    """
    from . import library
    root = Path(a.library or os.environ.get("NN_LIBRARY") or st.path.resolve().parent / "neuron-library")
    card = paper["card"] and load_card(st, paper["card"])
    scores = pole_scores(a, st, paper["neuron"])
    other_wins = bool(winning_runs(st, paper["neuron"]) - {paper["run"]})
    page, action = library.save(root, st.meta["model"], paper, card, scores, show, other_wins=other_wins)
    run = paper["run"] or "no run label"
    if action == "written":
        return f"library page {page}"
    if action == "removed":
        return f"no winning pole in {run}, removed library page {page}"
    if action == "kept":
        owner = library.page_run(page)
        if owner:
            return f"kept the library page of run {owner}: {run} has no winning pole"
        return f"kept the existing library page: {run} has no winning pole"
    return f"no winning pole, so no library page ({page})"


def cmd_library(a):
    st = open_store(a)
    papers = {n: p for (n,), p in newest(a, paper_dir(st), "neuron").items()}
    if missing := [n for n in a.neurons if n not in papers]:
        raise ValueError(f"no paper on {', '.join(missing)}" + (f" in run {a.run}" if a.run else ""))
    shows = showcase_map(a, st)
    for n in a.neurons or sorted(papers):
        print(f"{n}: {library_update(a, st, papers[n], shows.get(n))}")


# --- workflow commands -------------------------------------------------------

def cmd_select(a):
    st = open_store(a)
    alive = st.alive(DEAD_EPS, MIN_FIRES)
    cands = np.argwhere(alive)
    if a.top is not None:
        fires = st.fires[cands[:, 0], cands[:, 1]].sum(1)
        cands = cands[np.argsort(-fires, kind="stable")[:a.top]]
    if a.n > len(cands):
        raise ValueError(f"only {len(cands)} live neurons to choose from")
    pick = np.random.default_rng(a.seed).choice(len(cands), a.n, replace=False)
    ids = [f"L{l}.N{n}" for l, n in sorted(cands[pick].tolist())]
    if a.json:
        return emit_json(ids)
    pool = f", drawn from the top {a.top} by total fires" if a.top is not None else ""
    print(f"# {len(ids)} of {len(cands)} live neurons (of {alive.size}){pool}, seed {a.seed}", file=sys.stderr)
    print("\n".join(ids))


def score_one(a, st, test):
    """Score one blind test; see the score subcommand's help for the input format."""
    if "card" in test:
        test.setdefault("neuron", load_card(st, test["card"])["neuron"])
    neuron, pole = test["neuron"], test["pole"]  # required: the pole the tester was given
    if pole not in POLE_REGIMES:
        raise ValueError(f"pole must be + or -, not {pole!r}")
    layer, idx = parse_neuron(neuron)
    st.check(layer, idx)
    cmax = st.corpus_max(layer, idx)[pole]
    ref = short_max(a, st, neuron, layer, idx, pole)
    blind = test.get("blind_max", [])
    sets = {"blind_max": [blind] if isinstance(blind, str) else blind, **{s: test.get(s, []) for s in SETS}}
    texts = [t for ts in sets.values() for t in ts]
    acts = iter(post(a, "/run", neuron=neuron, texts=texts)) if texts else iter(())
    res = {"neuron": neuron, "pole": pole, "card": test.get("card"), "tester": test.get("tester"),
           "run": test.get("run", a.run), "corpus_max": cmax, "short_max": ref, "sets": {},
           "top": top_runs(a, st, neuron, layer, idx, pole)}
    for name, ts in sets.items():
        res["sets"][name] = [{"text": t, "act": (v := pole_value((run := next(acts))["acts"], pole)),
                              "ratio": v / ref if ref > 0 else 0.0,
                              "corpus_ratio": v / cmax if cmax > 0 else 0.0, **token_acts(run)} for t in ts]
    return judge(res)


def token_acts(run):
    """A /run result's tokens and signed activations, rounded, without the separator: what the library shades."""
    return {"tokens": run["tokens"][1:], "acts": [round(v, 3) for v in run["acts"][1:]]}


def top_runs(a, st, neuron, layer, idx, pole):
    """The pole's strongest corpus windows, each cut at its peak and run alone as in short_max, with per-token acts."""
    b = st.meta["before"]
    rows = st.examples(layer, idx, POLE_REGIMES[pole])[:PAPER_TOP]
    runs = post(a, "/run", neuron=neuron, ids=[[int(i) for i in w[:b + 1] if i >= 0] for _, _, w in rows])
    return [{"value": v, "regime": REGIMES[g], **token_acts(r)} for (v, g, _), r in zip(rows, runs)]


def fisher(a, n, b, m):
    """One-sided Fisher exact p-value that a of n is a higher rate than b of m.

    The chance of at least a strong texts among the n when the a + b strong ones are spread
    at random over all n + m texts (a hypergeometric tail).
    """
    k = a + b
    return sum(comb(k, x) * comb(n + m - k, n - x) for x in range(a, min(k, n) + 1)) / comb(n + m, n)


def judge(res):
    """Add strong rates, Fisher p-values, blind max and verdict to a result from its scored sets.

    WIN if new examples are strong more often than near misses with one-sided Fisher exact
    p < ALPHA (p_win), LOSS if near misses are strong more often with p < ALPHA (p_loss), else TIE.
    """
    ref = res["short_max"]
    strong = {s: (sum(r["ratio"] >= STRONG for r in rows), len(rows)) for s in SETS if (rows := res["sets"][s])}
    best = max((r["act"] for r in res["sets"]["blind_max"]), default=None)
    tested = "new" in strong and "near_miss" in strong
    p_win = fisher(*strong["new"], *strong["near_miss"]) if tested else None
    p_loss = fisher(*strong["near_miss"], *strong["new"]) if tested else None
    res.update(rate={s: strong[s][0] / strong[s][1] if s in strong else None for s in SETS},
               p_win=p_win, p_loss=p_loss,
               blind_ratio=None if best is None else best / ref if ref > 0 else 0.0,
               blind_pass=None if best is None else best > ref,
               verdict=None if not tested else "WIN" if p_win < ALPHA else "LOSS" if p_loss < ALPHA else "TIE")
    return res


def pool(results):
    """One result over several testers' tests of the same neuron pole and card: all their texts judged together."""
    labels = [r["tester"] or str(i) for i, r in enumerate(results, 1)]
    res = {k: results[0].get(k) for k in ("neuron", "pole", "card", "run", "corpus_max", "short_max", "top")}
    res["sets"] = {s: [{**x, "tester": t} for r, t in zip(results, labels) for x in r["sets"][s]]
                   for s in results[0]["sets"]}
    res["testers"] = [{"tester": t, **{k: r[k] for k in ("verdict", "rate", "p_win", "p_loss", "blind_ratio",
                                                         "blind_pass")}}
                      for r, t in zip(results, labels)]
    return judge(res)


def verdict_text(r):
    """The verdict, and for a pooled result each tester's own, e.g. WIN (WIN, TIE, WIN)."""
    v = r["verdict"] or "-"
    return f"{v} ({', '.join(t['verdict'] or '-' for t in r['testers'])})" if r.get("testers") else v


VERDICT_RULE = (f"A verdict is WIN if new examples are strong more often than near misses with one-sided Fisher "
                f"exact p < {ALPHA}, LOSS if near misses are strong more often with p < {ALPHA}, and TIE otherwise")


def strong_text(r):
    """The strong counts of a scored pole and the Fisher p-values of its verdict, as a sentence."""
    def count(s):
        rows = r["sets"][s]
        return f"{sum(x['ratio'] >= STRONG for x in rows)}/{len(rows)}" if rows else "-"
    return (f"Strong: new {count('new')}, near miss {count('near_miss')}, distant {count('distant')}. One-sided "
            f"Fisher exact p = {pval(r.get('p_win'))} for new above near miss, {pval(r.get('p_loss'))} for near "
            f"miss above new.")


def summary_line(r):
    def rate(s):
        rows = r["sets"][s]
        return f"{s} {sum(x['ratio'] >= STRONG for x in rows)}/{len(rows)}" if rows else f"{s} -"
    blind = ("blind max -" if r["blind_ratio"] is None else
             f"blind max {pct(r['blind_ratio'], 1)} {'PASS' if r['blind_pass'] else 'fail'}")
    who = f"pooled over {len(r['testers'])} testers" if r.get("testers") else f"tester {r.get('tester') or '-'}"
    return (f"{r['neuron']} {r['pole']} {who} | short-context max {r['short_max']:.3f} (corpus "
            f"{r['corpus_max']:.3f}) | {blind} | strong: {', '.join(rate(s) for s in SETS)} | "
            f"Fisher p new>near {pval(r.get('p_win'))}, near>new {pval(r.get('p_loss'))} | {verdict_text(r)}")


def cmd_score(a):
    st = open_store(a)
    score_dir(st).mkdir(exist_ok=True)

    def save(r, tag=""):
        ns = time.time_ns()
        r["time"] = ns / 1e9
        name = f"{r['neuron']}_{'pos' if r['pole'] == '+' else 'neg'}_{r['card'] or 'nocard'}{tag}_{ns}.json"
        (score_dir(st) / name).write_text(json.dumps(r, indent=1))
        return r

    results = []
    for path in a.files:
        data = json.loads(Path(path).read_text())
        results += [save(score_one(a, st, test)) for test in (data if isinstance(data, list) else [data])]
    groups = {}
    for r in results:
        groups.setdefault((r["neuron"], r["pole"], r["card"], r["run"]), []).append(r)
    results += [save(pool(g), "_pooled") for g in groups.values() if len(g) > 1]  # newest, so the report uses it
    for r in results:
        print(summary_line(r))


def cmd_report(a):
    st = open_store(a)
    results = [r for _, r in sorted(newest(a, score_dir(st), "neuron", "pole").items())]  # newest score per pole
    latest = newest(a, card_dir(st), "neuron")
    papers = {n: p for (n,), p in newest(a, paper_dir(st), "neuron").items()}
    shows = showcase_map(a, st)
    neurons = sorted({r["neuron"] for r in results} | set(papers))

    def card_of(r):
        return load_card(st, r["card"]) if r.get("card") else latest.get((r["neuron"],))

    def pole_section(r):
        c = card_of(r)
        out = ["", f"### {r['pole']} pole: {verdict_text(r) if r['verdict'] else 'not scored'}", ""]
        out += [card_text(c).replace("\n", "\n\n"), ""] if c else ["No card.", ""]
        if blind := c and c.get("redacted", {}).get(r["pole"]):
            out += [f"Blind tester read: {blind}", ""]
        out += [f"Short-context max {r['short_max']:.3f}, corpus max {r['corpus_max']:.3f}. Values are % of the "
                f"short-context max; strong is >= {STRONG:.0%}.", "",
                "| Set | Text | Activation | % | % of corpus max |", "|---|---|---|---|---|"]
        for s, rows in r["sets"].items():
            out += [f"| {s}{' (' + cell(x['tester']) + ')' if 'tester' in x else ''} | {cell(x['text'], 160)} | "
                    f"{x['act']:.3f} | {percent(x['ratio'])} | {percent(x['corpus_ratio'])} |" for x in rows]
        return out + ["", strong_text(r)]

    verdicts = [r["verdict"] for r in results]
    passes = [r["blind_pass"] for r in results if r["blind_pass"] is not None]
    mean = {s: [r["rate"][s] for r in results if r["rate"][s] is not None] for s in SETS}
    pooled = sum(bool(r.get("testers")) for r in results)
    out = ["# Blind neuron explanation report", "",
           (f"Run {a.run}. " if a.run else "") +
           f"Model {st.meta['model']}, {st.meta['tokens']:,} harvested tokens. {len(results)} neuron poles scored: "
           f"{verdicts.count('WIN')} win, {verdicts.count('TIE')} tie, {verdicts.count('LOSS')} loss. "
           f"Blind max beat the short-context max for {sum(passes)}/{len(passes)}. {VERDICT_RULE}." +
           (f" {pooled} poles had several blind testers: they are judged on all their texts together, "
            f"with each tester's own verdict in brackets." if pooled else ""),
           f"The short-context max is the pole's largest value over its stored top windows, "
           f"each cut at its peak and run alone. Mean strong rate (activation >= {STRONG:.0%} of it): " +
           ", ".join(f"{s} {percent(np.mean(v)) if v else '-'}" for s, v in mean.items()) + ".", ""]
    if papers:
        out += [f"Papers on {len(papers)} neurons, mean clean {np.mean([p['clean'] for p in papers.values()]):.1f}/5, "
                f"mean interesting {np.mean([p['interesting'] for p in papers.values()]):.1f}/5.", "",
                "| Neuron | Paper | Clean | Interesting | + verdict | - verdict |", "|---|---|---|---|---|---|"]
        verdict = {(r["neuron"], r["pole"]): verdict_text(r) for r in results}
        for n in neurons:
            p = papers.get(n, {})
            out.append(f"| {n} | {cell(p.get('title', '-'))} | {p.get('clean', '-')} | {p.get('interesting', '-')} | "
                       f"{verdict.get((n, '+'), '-')} | {verdict.get((n, '-'), '-')} |")
        out.append("")
    out += ["| Neuron | Pole | Explanation | Blind max | New | Near miss | Distant | p new>near | p near>new | "
            "Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = card_of(r)
        expl = c and ((r["pole"] == "-" and c.get("negative")) or c["core"])
        out.append(f"| {r['neuron']} | {r['pole']} | {cell(expl, 90)} | "
                   f"{percent(r['blind_ratio'])} | {percent(r['rate']['new'])} | {percent(r['rate']['near_miss'])} | "
                   f"{percent(r['rate']['distant'])} | {pval(r.get('p_win'))} | {pval(r.get('p_loss'))} | "
                   f"{verdict_text(r)} |")
    for n in neurons:
        out += ["", f"## {n}: {papers[n]['title']}" if n in papers else f"## {n}"]
        if p := papers.get(n):
            out += ["", p["summary"], "", f"Clean {p['clean']}/5: {p['clean_why']}", "",
                    f"Interesting {p['interesting']}/5: {p['interesting_why']}"]
            for pole, rows in p["contrast"].items():
                out += ["", f"Contrast, {pole} pole: new texts beside the near miss written with them "
                        f"(% of the short-context max).", "", "| New text | % | Near miss | % |",
                        "|---|---|---|---|"] if rows else []
                out += [f"| {cell(x['new'], 120)} | {percent(x['new_ratio'])} | {cell(x['near_miss'], 120)} | "
                        f"{percent(x['near_miss_ratio'])} |" for x in rows]
        for r in results:
            if r["neuron"] == n:
                out += pole_section(r)
    if shows:
        out += showcase_md(shows, a.run)
    print("\n".join(out) + "\n", end="")


# --- operator commands -------------------------------------------------------

def benign_source(a):
    """The benign contrast source for `nn screen`."""
    from .textsource import make_source
    where = {}
    for item in a.benign_where:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--benign-where needs KEY=VALUE, got {item!r}")
        where[key] = value
    field = a.benign_field or a.field
    fields = [f.strip() for f in field.split(",")] if field else None
    dataset = a.benign_dataset or a.dataset
    if dataset is None and a.benign_texts is None:
        raise ValueError("screen needs a benign contrast: --benign-dataset/--benign-where or --benign-texts")
    if a.benign_dataset is None and not where and a.benign_texts is None:
        print("nn screen: warning: no --benign-where, so the benign source is the same as the positive source",
              file=sys.stderr)
    return make_source(dataset=dataset,
                       config=a.benign_config if a.benign_dataset else a.config,
                       split=a.benign_split if a.benign_dataset else a.split,
                       fields=fields, where=where, texts=a.benign_texts)


def cmd_screen(a):
    from .screen import screen, write
    res = screen(a.model, get_source(a), benign_source(a), intent=a.intent,
                 max_texts=a.max_texts, tokens=a.tokens, top_k=a.top_k, batch=a.batch,
                 device=a.device, match=a.match, benign_match=a.benign_match,
                 min_mean=a.min_mean, max_docs=a.max_docs, benign_max_docs=a.benign_max_docs)
    write(a.out, res)
    print(f"screened {res['texts']['positive']} positive and {res['texts']['negative']} benign texts "
          f"({res['texts']['tokens']} content tokens each); top {len(res['neurons'])} -> {a.out}")
    print(f"  {'neuron':<10} {'t':>6}  {'mean pos':>8} {'max pos':>8}  {'mean neg':>8} {'max neg':>8}  peak")
    for r in res["neurons"][:15]:
        print(f"  {r['neuron']:<10} {r['t']:6.2f}  {r['mean_pos']:8.2f} {r['max_pos']:8.1f}  "
              f"{r['mean_neg']:8.2f} {r['max_neg']:8.1f}  {r['peak_pos']!r}")


def cmd_rank(a):
    from .rank import rank
    i, _, n = a.shard.partition("/")
    res = rank(a.out, a.model, a.tokens, k=a.k, fire=a.fire, ctx=a.ctx, batch=a.batch,
               checkpoint=a.checkpoint, shard=(int(i), int(n)), device=a.device,
               source=get_source(a), max_docs=a.max_docs, stat=a.stat)
    stat = res["stat"]
    print(f"{res['tokens']:,} tokens of {res['model']} at fire {res['fire']} over {res['docs']:,} documents; "
          f"top {len(res['neurons'])} neurons by {stat}")
    print(f"rank, neuron, + {stat}, - {stat}, total")
    fmt = "{:>10,}" if stat == "fires" else "{:>10.4f}"
    for r, (neuron, (plus, minus)) in enumerate(zip(res["neurons"], res["counts"]), 1):
        print(f"{r:3d} {neuron:<10} {fmt.format(plus)} {fmt.format(minus)} {fmt.format(plus + minus)}")
    sys.stdout.flush()
    os._exit(0)  # skip slow teardown of the dataset stream


def cmd_harvest(a):
    from .harvest import harvest
    if not a.store:
        raise ValueError("no store: set NN_STORE or pass --store")
    neurons = None
    if a.neurons:
        path = Path(a.neurons)
        if not path.exists():
            raise ValueError(f"no neuron file {a.neurons}")
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise ValueError(f"bad neuron file {a.neurons}: {e}") from None
        neurons = data.get("neurons") if isinstance(data, dict) else data
        if not isinstance(neurons, list) or not all(isinstance(s, str) for s in neurons):
            raise ValueError(f'{a.neurons}: expected a list of neuron ids or {{"neurons": [...]}}')
        for s in neurons:
            parse_neuron(s)
    harvest(a.store, a.model, a.tokens, ctx=a.ctx, batch=a.batch, k=a.topk, window=a.window,
            before=a.before, fire=a.fire, checkpoint=a.checkpoint, device=a.device,
            neurons=neurons, source=get_source(a), max_docs=a.max_docs)
    os._exit(0)  # skip slow teardown of the dataset stream


def cmd_merge(a):
    from .harvest import merge_stores
    merge_stores(a.out, a.inputs)
    print(f"merged {len(a.inputs)} stores into {a.out}")


def cmd_serve(a):
    from .server import serve
    model = a.model or open_store(a).meta["model"]
    url = urlparse(a.server)
    serve(model, host=url.hostname, port=url.port or 80, device=a.device)


# --- argument parsing ----------------------------------------------------------

def library_arg(s):
    s.add_argument("--library", help="neuron library root; pages go to <root>/<model name>/<neuron>.html and the "
                                     "index to <root>/index.html (default $NN_LIBRARY, else neuron-library/ beside "
                                     "the store)")


def source_args(s):
    s.add_argument("--dataset", help="Hugging Face dataset id to stream instead of FineWeb, "
                                     "e.g. PKU-Alignment/BeaverTails")
    s.add_argument("--config", help="dataset config name (default: the dataset's default config)")
    s.add_argument("--split", help="dataset split (default train; e.g. 330k_train for BeaverTails)")
    s.add_argument("--field", help="comma-separated column(s) to read as the text (default text); "
                                   "multiple columns are joined with newlines")
    s.add_argument("--where", action="append", default=[], metavar="KEY=VALUE",
                   help="keep only rows whose column equals the value, case-insensitively; repeatable, "
                        "e.g. --where is_safe=false")
    s.add_argument("--texts", help="a JSONL or plain-text file to read instead of a dataset")
    s.add_argument("--max-docs", type=int, default=0, help="stop after this many documents (0 = all)")


def get_source(a):
    """The text source spec from the --dataset/--texts options (nn.textsource)."""
    from .textsource import make_source
    where = {}
    for item in a.where:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--where needs KEY=VALUE, got {item!r}")
        where[key] = value
    fields = [f.strip() for f in a.field.split(",")] if a.field else None
    return make_source(dataset=a.dataset, config=a.config, split=a.split,
                       fields=fields, where=where, texts=a.texts)


def parser():
    p = argparse.ArgumentParser(prog="nn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default=os.environ.get("NN_STORE"), help="store directory (default $NN_STORE)")
    p.add_argument("--server", default=os.environ.get("NN_SERVER", "http://127.0.0.1:8765"),
                   help="neuron server URL (default $NN_SERVER or http://127.0.0.1:8765)")
    p.add_argument("--run", default=os.environ.get("NN_RUN"),
                   help="run label (default $NN_RUN): stamped on new records and limits card, paper, library "
                        "and report lookups to that run")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help, neuron=False, json=False):
        s = sub.add_parser(name, help=help, description=help)
        if fn:
            s.set_defaults(func=fn)
        if neuron:
            s.add_argument("neuron", help="e.g. L12.N3456")
        if json:
            s.add_argument("--json", action="store_true", help="machine-readable output")
        return s

    s = add("top", cmd_top, "strongest corpus examples of a neuron pole", neuron=True)
    s.add_argument("--pole", choices="+-", help="default + (or - for a mixed regime)")
    s.add_argument("--regime", choices=list(REGIME_ARGS), help="gate x up signs; mixed = +- or -+")
    s.add_argument("-n", type=int, default=10, help="examples to show (default 10)")
    s.add_argument("--start", type=int, default=0, help="first rank to show")

    s = add("run", cmd_run, "per-token activations of a neuron on any text", neuron=True)
    s.add_argument("--text", required=True)
    s.add_argument("--min", type=float, default=0.1, help="annotate tokens beyond this share of corpus max")

    for name, fn, help in (("steer", cmd_steer, "next-token changes when the activation is multiplied by --scale"),
                           ("ablate", cmd_ablate, "next-token and logit changes when the neuron is zeroed")):
        s = add(name, fn, help, neuron=True)
        if name == "steer":
            s.add_argument("--scale", type=float, required=True)
        s.add_argument("--text", required=True)
        s.add_argument("--top", type=int, default=8, help="tokens per list")

    help = ("generate one baseline completion and one per calibrated intervention. --add/--set strengths are in "
            "sigma units by default (--ref sigma|absmax|resid), --scale is a plain multiplier. Checks whether "
            "forcing a documented neuron produces coherent, on-concept or unaligned behaviour")
    s = add("generate", cmd_generate, help, neuron=True, json=True)
    s.add_argument("--text", required=True)
    s.add_argument("--chat", action="store_true", help="wrap --text as one user message in the chat template")
    s.add_argument("--add", type=float_list, default=[64.0, 128.0, -64.0, -128.0],
                   help="strengths for h += strength*unit (default 64,128,-64,-128; negative = - pole)")
    s.add_argument("--set", type=float_list, default=[64.0, 128.0, -64.0, -128.0],
                   help="strengths for h = strength*unit (default 64,128,-64,-128)")
    s.add_argument("--scale", type=float_list, default=[2.0, 3.0],
                   help="multipliers for h *= strength (default 2,3)")
    s.add_argument("--ref", choices=list(STEER_REFS), default="sigma",
                   help="unit for --add/--set: sigma = prompt std (default), absmax = prompt absmax, resid = "
                        "mean residual norm / down_proj column norm")
    s.add_argument("--positions", choices=list(STEER_POSITIONS), default="all",
                   help="all positions, only generated tokens, or only the prompt (default all)")
    s.add_argument("--seed", type=int, default=0, help="sampling seed, reset before BASE and every run")
    s.add_argument("--max-new-tokens", type=int, default=80)
    s.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    s.add_argument("--model", default="Qwen/Qwen3-0.6B")
    s.add_argument("--device", default="cuda")

    s = add("attrib", cmd_attrib, "upstream neurons and heads driving the neuron (gradient x activation)", neuron=True)
    s.add_argument("--example", required=True, help="rank from `nn top` (same --pole) or a text")
    s.add_argument("--pole", choices="+-", default="+")
    s.add_argument("--top", type=int, default=15, help="units per list")

    s = add("card", None, "submit, redact or read explanation cards", json=False)
    card = s.add_subparsers(dest="card_cmd", required=True)
    c = card.add_parser("submit", help=f"save a card and print its id; each field at most {MAX_FIELD} characters")
    c.set_defaults(func=cmd_card_submit)
    c.add_argument("neuron")
    c.add_argument("--core", required=True, help="core explanation, one sentence")
    c.add_argument("--negative", help="explanation of the negative pole")
    c.add_argument("--offshoots", help="known offshoots")
    c.add_argument("--residue", help="unexplained residue")
    help = ("store a pole's anonymized text, the only thing its blind tester reads: abstract words, no examples, "
            "numbers or unit ids; once per pole")
    c = card.add_parser("redact", help=help, description=help)
    c.set_defaults(func=cmd_card_redact)
    c.add_argument("card_id")
    c.add_argument("--pole", choices="+-", required=True)
    c.add_argument("--text", required=True, help=f"at most {MAX_FIELD} characters")
    c = card.add_parser("read", help="print a card's full text, or with --pole only that pole's anonymized text")
    c.set_defaults(func=cmd_card_read)
    c.add_argument("card_id")
    c.add_argument("--pole", choices="+-", help="what this pole's blind tester reads; fails if not redacted yet")

    s = add("steering", None, "save shareable steering showcase records", json=False)
    steer_sub = s.add_subparsers(dest="steering_cmd", required=True)
    c = steer_sub.add_parser("save", help="validate one JSON record from stdin or --file, write it under "
                              "<store>/steering/ and print the path")
    c.set_defaults(func=cmd_steering_save)
    c.add_argument("--file", help="read the record from this file (default stdin; - = stdin)")

    help = ("with NEURON, print its paper facts (the card its newest score tested, and per pole the newest score, "
            "top examples and contrast pairs). With save NEURON, store the paper and update its library page; the "
            "page also shows the newest steering showcase record of the run")
    s = add("paper", cmd_paper, help)
    s.add_argument("what", metavar="NEURON|save", help="a neuron, to print its facts; or save")
    s.add_argument("neuron", nargs="?", help="with save: the neuron the paper is on")
    s.add_argument("--title", help=f"save: at most {MAX_TITLE} characters")
    s.add_argument("--summary", help=f"save: the paper's paragraph, at most {MAX_PARAGRAPH} characters")
    for k in ("clean", "interesting"):
        s.add_argument(f"--{k}", type=int, choices=range(1, 6), help="save: score 1 (low) to 5 (high)")
        s.add_argument(f"--{k}-why", help=f"save: the reason for the {k} score, at most {MAX_FIELD} characters")
    library_arg(s)

    help = ("rebuild: re-render the pages of the given neurons, or of every neuron with a paper (with --run, in "
            "that run), from the newest paper, scores and steering record; writes no new records. A page exists only "
            "for a winning pole and shows only winning poles; rebuilding a run with no winning pole never removes "
            "another run's page; the index is rewritten each time")
    s = add("library", cmd_library, help, json=False)
    s.add_argument("action", choices=["rebuild"])
    s.add_argument("neurons", nargs="*", help="e.g. L12.N3456; default all with a paper")
    library_arg(s)

    s = add("select", cmd_select, "sample random live (non-dead) harvested neurons", json=True)
    s.add_argument("n", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--top", nargs="?", const=20, type=int,
                   help="draw from the K alive neurons with the most fires (default 20 when given)")

    s = add("score", cmd_score, 'score blind tests. Each file holds one test or a list: {"neuron": "L12.N3456" '
            'or "card": id, "pole": "+" or "-" (required), "blind_max": text or [texts], "new": [...], '
            '"near_miss": [...], "distant": [...], "tester": optional label, "run": optional label (default --run)}. '
            'Each test adds a file to $NN_STORE/scores. Several tests of the same neuron, pole, card and run label '
            'in one call (one per blind tester) also add a pooled result, judged on all their texts together; the '
            f'report shows the newest per pole. {VERDICT_RULE}, for each test and for the pooled result.')
    s.add_argument("files", nargs="+")

    s = add("report", cmd_report, "markdown report of the scored neurons, their cards, papers and the run's steering "
            "showcase (with --run, of that run only)", json=False)

    help = ("rank neurons by how specifically an intent corpus activates them over a benign contrast. Reads "
            "equal-length token prefixes from both sources, skips the separator and the first content token, and "
            "writes JSON with a Welch t, per-class means and maxima, and the peak token of each pole. Harvest the "
            "top neurons with `nn harvest --neurons`.")
    s = add("screen", cmd_screen, help)
    s.add_argument("--model", default="Qwen/Qwen3-0.6B")
    s.add_argument("--intent", default="", help="name of the intent, copied into the output")
    s.add_argument("--out", required=True, help="write the ranked JSON here")
    s.add_argument("--top-k", type=int, default=60, help="neurons to write (default 60)")
    s.add_argument("--max-texts", type=int, default=120, help="texts per class (default 120)")
    s.add_argument("--tokens", type=int, default=100, help="content tokens per text (default 100)")
    s.add_argument("--batch", type=int, default=16, help="texts per forward pass")
    s.add_argument("--device", default="cuda")
    s.add_argument("--match", help="regex the positive text must match")
    s.add_argument("--benign-match", help="regex the benign text must match")
    s.add_argument("--min-mean", type=float, default=0.5, help="drop neurons whose positive mean is below this")
    source_args(s)
    s.add_argument("--benign-dataset", help="benign dataset id (default: the positive dataset)")
    s.add_argument("--benign-config")
    s.add_argument("--benign-split")
    s.add_argument("--benign-field")
    s.add_argument("--benign-where", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--benign-texts")
    s.add_argument("--benign-max-docs", type=int, default=0)

    s = add("rank", cmd_rank, "rank neurons by total fires over a text corpus (FineWeb by default; "
            "--dataset/--texts for another); writes <out>/top.json for `nn harvest --neurons`")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--model", default="Qwen/Qwen3-0.6B")
    s.add_argument("--tokens", type=float, default=1e8, help="tokens to stream")
    s.add_argument("--k", type=int, default=20, help="neurons written to top.json")
    s.add_argument("--stat", choices=["fires", "max", "mean", "spike"], default="fires",
                   help="ranking statistic: fires = tokens above --fire; max = peak magnitude; "
                        "mean = mean positive activation; spike = max - mean (neurons these texts trigger)")
    s.add_argument("--fire", type=float, default=0.1, help="|h| counted as a fire (fires stat)")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--batch", type=int, default=8, help="sequences per forward pass")
    s.add_argument("--checkpoint", type=float, default=5e7, help="save every this many tokens")
    s.add_argument("--shard", default="0/1", help="i/n: this process's share of the stream")
    s.add_argument("--device", default="cuda")
    source_args(s)

    s = add("harvest", cmd_harvest, "stream a text corpus (FineWeb by default; --dataset/--texts) and build "
            "the store (resumes if the store exists)", json=False)
    s.add_argument("--model", default="Qwen/Qwen3-0.6B")
    s.add_argument("--tokens", type=float, default=1e8, help="total tokens for this shard")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--batch", type=int, default=8, help="sequences per forward pass")
    s.add_argument("--topk", type=int, default=20, help="examples kept per neuron and regime")
    s.add_argument("--window", type=int, default=32, help="tokens kept around each peak")
    s.add_argument("--before", type=int, default=24, help="of which before the peak")
    s.add_argument("--fire", type=float, default=0.1, help="|h| counted as a fire")
    s.add_argument("--checkpoint", type=float, default=5e7, help="save every this many tokens")
    s.add_argument("--shard", default="0/1", help="i/n: this process's share of the stream")
    s.add_argument("--device", default="cuda")
    s.add_argument("--neurons", help="JSON file of neuron ids (a list or {\"neurons\": [...]}), e.g. a rank "
                                     "top.json or a screen picks file: harvest only those neurons")
    source_args(s)

    s = add("merge", cmd_merge, "combine stores harvested on different shards", json=False)
    s.add_argument("out")
    s.add_argument("inputs", nargs="+")

    s = add("serve", cmd_serve, "load the model and serve the API at --server", json=False)
    s.add_argument("--model", help="default: the store's model")
    s.add_argument("--device", default="cuda")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--regime" in argv[:-1]:  # let `--regime --` through argparse, which reads "--" as end of options
        i = argv.index("--regime")
        argv[i:i + 2] = [f"--regime={argv[i + 1]}"]
    a = parser().parse_args(argv)
    try:
        a.func(a)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"nn: {e}")
    except KeyError as e:
        sys.exit(f"nn: missing field {e}")


if __name__ == "__main__":
    main()
