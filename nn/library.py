"""The neuron library: an HTML page per neuron whose blind test was won, and an index of the pages.

<root>/<model name>/<neuron>.html shows a paper (title, paragraph, clean and interesting scores), the run's newest
steering showcase record when there is one (query, full BASE and steered generation, control, the judge's
classification, recipe and reproduce command) and, for each pole whose newest score is a WIN, in reading order:
what its blind testers read, the verdict (a dot per text, filled when strong), a chart of each new text in full
beside the bars of it and its near-miss twin, and texts shaded token by token by activation in four steps with a
legend. Ties and losses stay in the store. A neuron with no winning pole has no page. <root>/index.html lists every page under the root, read back from the metadata
each page embeds, so it covers all runs and stores that write to the root.

Rebuilding one run must not delete another run's page: each page embeds its run label, and a
run-scoped `nn library rebuild` removes a page only when it owns it or when no other run still
has a winning score for that neuron (see save).

Pages are self-contained: templates/neuron.html and templates/index.html with templates/style.css inlined, filled
with string.Template; every text is escaped, and the charts are plain HTML so labels wrap instead of being cut.
"""

import html
import json
import math
import re
import time
from pathlib import Path
from string import Template

from .cli import ALPHA, PAPER_PAIRS, STRONG, percent, pval, twins
from .showcase import dose_md

TEMPLATES = Path(__file__).parent / "templates"
POLE_NAMES = {"+": "Positive pole", "-": "Negative pole"}
CARD_LABELS = {"core": "Core", "negative": "Negative pole", "offshoots": "Known offshoots",
               "residue": "Unexplained residue"}
BINS = (0.05, STRONG, 0.6, 1.0)  # shading steps: share of the short-context max where each step starts
BIN_LABELS = ("5–30%", "30–60%", "60–100%", "100%+")
META = re.compile(r'<script type="application/json" id="nn-page">(.*?)</script>', re.S)


def esc(s):
    return html.escape(str(s))


def render(name, **fields):
    return Template((TEMPLATES / name).read_text()).substitute(style=(TEMPLATES / "style.css").read_text(), **fields)


def sign(pole):
    return "−" if pole == "-" else "+"


def raw(v, pole):
    """A raw activation on the pole, as its signed value: the store keeps magnitudes."""
    return f"{sign(pole)}{v:.2f}"


def p_text(p):
    s = pval(p)
    return f"p {s[0]} {s[1:]}" if s.startswith("<") else f"p = {s}"


def verdict_chip(v):
    return f'<span class="chip {(v or "none").lower()}">{v or "–"}</span>'


def pips(n, label):
    dots = "".join(f'<i class="{"on" if i < n else ""}"></i>' for i in range(5))
    return f'<span class="pips" role="img" aria-label="{label} {n} of 5">{dots}</span>'


def page_meta(path):
    """The metadata a page embeds, or None when the file is not one of this module's pages."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    m = META.search(text)
    if not m:
        return None
    try:
        return json.loads(m[1])
    except ValueError:
        return None


def page_run(path):
    """The run label embedded in a page, or None when it has none or is not a page we wrote."""
    meta = page_meta(path)
    return meta.get("run") if meta else None


def may_remove(page, run, other_wins):
    """Whether rebuilding `run` may delete this page: it embeds that same run, or no other run wins.

    A file without readable page metadata is not a page this library wrote, so it is never removed.
    """
    meta = page_meta(page)
    if meta is None:
        return False
    return meta.get("run") == run or not other_wins


def save(root, model, paper, card, scores, show=None, other_wins=False):
    """Write the paper's page if a pole's newest score is a WIN, else remove it only when that is safe.

    scores holds the newest score per pole of this run (all runs when the paper has no run label);
    show is the newest steering showcase record, or None. `other_wins` says whether another run
    has a winning score for the neuron.

    A run-scoped rebuild must never delete a page another run produced. When this run has no
    winning pole, an existing page is removed only if it embeds this run's label or if no other
    run has a winning score; otherwise it is left untouched.

    Returns (path, action), with action "written", "removed", "kept" (another run owns the page)
    or "absent" (no page and no winning pole). The index is rewritten in every case.
    """
    page = Path(root) / model.split("/")[-1] / f"{paper['neuron']}.html"
    won = {pole: r for pole, r in sorted(scores.items()) if r.get("verdict") == "WIN"}
    if won:
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(neuron_page(model, paper, card, won, show))
        action = "written"
    elif not page.exists():
        action = "absent"
    elif not may_remove(page, paper["run"], other_wins):
        action = "kept"
    else:
        page.unlink()
        action = "removed"
    write_index(Path(root))
    return page, action


# --- the neuron page -----------------------------------------------------------

def neuron_page(model, paper, card, won, show=None):
    meta = {"neuron": paper["neuron"], "model": model, "title": paper["title"], "poles": list(won),
            "clean": paper["clean"], "interesting": paper["interesting"], "run": paper["run"], "time": paper["time"]}
    pairs = {p: twins(r) for p, r in won.items()}
    top = max([1.0] + [x["ratio"] for ps in pairs.values() for p in ps for x in p])
    hi = math.ceil(top * 2) / 2  # one chart scale for every pole on the page, ending on a multiple of 50%
    jump = "".join(f'<li><a href="#{anchor(p)}">{POLE_NAMES[p]}</a> <span class="chip win">WIN</span></li>'
                   for p in won)
    if show:
        jump += '<li><a href="#showcase">Showcase</a></li>'
    return render("neuron.html", title=esc(paper["title"]), neuron=esc(paper["neuron"]), model=esc(model),
                  model_short=esc(model.split("/")[-1]), run=esc(paper["run"] or "-"),
                  card_id=esc(paper["card"] or "-"),
                  date=time.strftime("%Y-%m-%d %H:%M", time.localtime(paper["time"])), summary=esc(paper["summary"]),
                  clean=paper["clean"], clean_pips=pips(paper["clean"], "clean"), clean_why=esc(paper["clean_why"]),
                  interesting=paper["interesting"], interesting_pips=pips(paper["interesting"], "interesting"),
                  interesting_why=esc(paper["interesting_why"]), jump=jump,
                  poles="\n".join(pole_section(p, r, card, pairs[p], hi) for p, r in won.items()),
                  showcase=showcase_section(show), card=card_details(card), strong=f"{STRONG:.0%}", alpha=ALPHA,
                  meta=json.dumps(meta).replace("<", "\\u003c"))


def anchor(pole):
    return "pole-pos" if pole == "+" else "pole-neg"


def showcase_section(show):
    """The run's shareable generation in full, with its control, the judge's verdict, recipe and command.

    Everything comes escaped from the store record; the dose label comes from the same helper the markdown
    report uses, so both pages say the same thing.
    """
    if not show:
        return ""
    c = show["classification"]
    meta = f"run {show['run']} · {show['model']}"
    settings = " · ".join(["chat template" if show["chat"] else "raw prompt",
                           f"temperature {show['temperature']:g}", f"seed {show['seed']}"])
    chips = (f'<span class="chip {"win" if c["coherent"] else "loss"}">coherent '
             f'{"yes" if c["coherent"] else "no"}</span> '
             f'<span class="chip {"win" if c["on_concept"] else "tie"}">on-concept '
             f'{"yes" if c["on_concept"] else "no"}</span> '
             f'<span class="chip {"loss" if c["unaligned"] else "tie"}">unaligned '
             f'{"yes" if c["unaligned"] else "no"}</span> '
             f'<span class="chip tie">movement {c["movement"]}/3</span>')
    notes = []
    if c["note"]:
        notes.append(c["note"])
    if not c["coherent"]:
        notes.append("The judge called this run incoherent; it is shown for its concept shift, not as a usable answer.")
    if c["unaligned"]:
        notes.append("The judge marked it unaligned.")

    def gen(label, detail, text, cls):
        return (f'<div class="genbox"><div class="genhead"><span class="kind">{esc(label)}</span>'
                f'<span class="genv">{detail}</span></div>'
                f'<p class="gen {cls}">{esc(text)}</p></div>')

    out = ['<section class="showcase" id="showcase" aria-labelledby="showcase-h">',
           f'<div class="part">{esc(meta)}</div>',
           '<h2 id="showcase-h">Showcase: a steered generation</h2>',
           '<p class="lede-s">One generation from the run\'s calibrated dose probe. The judge\'s classification is '
           'copied unchanged: a run it called incoherent is labelled that way and is evidence of what the neuron does, '
           'not a usable answer. The recipe and the command reproduce it exactly.</p>',
           '<div class="plate gens">',
           gen("Query", esc(settings), show["query"], "query"),
           gen("BASE", "clean prompt", show["base"], "base"),
           gen("STEERED", esc(dose_md(show, detail=True)), show["steered"], "steered")]
    control = show.get("control")
    if control:
        out.append(gen("CONTROL", esc(dose_md(control)), control["text"], "control"))
    else:
        out.append('<p class="note">No control recorded for this run.</p>')
    out += ['</div>',
            f'<div class="plate"><p class="vline">{chips}<span>{esc(" ".join(notes))}</span></p>',
            '<dl class="facts">',
            f'<div><dt>Why this run</dt><dd>{esc(show["selection"])}</dd></div>',
            f'<div><dt>Recipe</dt><dd>{esc(show["recipe"])}</dd></div>',
            '</dl>',
            '<h3>Reproduce</h3>',
            f'<pre class="repro"><code>{esc(show["reproduce"])}</code></pre>',
            '</div>',
            '</section>']
    return "\n".join(out)


def pole_section(pole, r, card, pairs, hi):
    """What the blind testers read, the verdict, the contrast chart, then the shaded texts."""
    ref = r["short_max"]
    blind = (card or {}).get("redacted", {}).get(pole)
    out = [f'<section class="pole" id="{anchor(pole)}" aria-labelledby="{anchor(pole)}-h">',
           f'<div class="part">{esc(r.get("neuron", ""))} · <b class="sg">{sign(pole)}</b> pole</div>',
           f'<h2 id="{anchor(pole)}-h">{POLE_NAMES[pole]} <span class="chip win">WIN</span></h2>']
    if blind:
        out += ['<h3>What its blind testers read</h3>', f'<blockquote class="blind"><p>{esc(blind)}</p></blockquote>']
    out += ['<h3>Verdict</h3>', verdict(r)]
    if pairs:
        out += ['<h3>Each new text against its near-miss twin</h3>',
                f'<p class="lede-s">Each row is a new text a tester wrote to fire the neuron (upper bar) and the near '
                f'miss written beside it not to (lower bar), widest gap first. Bars are percentages of the '
                f'short-context max, {raw(ref, pole)}: the pole\'s largest value over its top corpus windows, each cut '
                f'at its peak and run alone (corpus max {raw(r["corpus_max"], pole)}).</p>', contrast_chart(pairs, hi)]
    if r.get("top") or pairs:
        out += ['<h3>Texts, shaded by activation</h3>',
                '<p class="lede-s">Each token is shaded by its activation on this pole as a share of the '
                'short-context max, in four steps; tokens under 5% stay plain. The underlined token is the '
                'text\'s peak.</p>', ramp()]
    if r.get("top"):
        out += ['<h4>Top corpus windows, cut at the peak and run alone</h4>',
                f'<p class="lede-s">Each window is the text up to the token where the neuron peaked in its full '
                f'document. Run alone, without the earlier text, the value usually drops; the short-context max, '
                f'{raw(ref, pole)}, is the largest of these.</p>', '<div class="exs">']
        for i, x in enumerate(r["top"], 1):
            out.append(f'<div class="plate exbox">{example("corpus", f"#{i}", "", x, pole, ref, full=x["value"])}</div>')
        out.append("</div>")
    if pairs:
        out += ['<h4>The widest-gap twins</h4>', '<div class="exs">']
        for x, y in pairs[:PAPER_PAIRS]:
            who = f"tester {x['tester']}" if x.get("tester") else ""
            out += ['<div class="plate exbox twin">', example("new", "New text", who, x, pole, ref),
                    example("near", "Near-miss twin", who, y, pole, ref), "</div>"]
        out.append("</div>")
    return "\n".join(out + ["</section>"])


def verdict(r):
    """One line of verdict, a dot strip per text set (filled = strong), the testers and the blind max."""
    def strip(key, cls, name, what):
        rows = sorted(r["sets"].get(key) or [], key=lambda x: -x["ratio"])
        if not rows:
            return ""
        n = sum(x["ratio"] >= STRONG for x in rows)
        dots = "".join(f'<i class="{"on" if x["ratio"] >= STRONG else ""}"></i>' for x in rows)
        return (f'<div class="strip {cls}"><div class="sname">{name}<small>{what}</small></div>'
                f'<div class="dots" role="img" aria-label="{n} of {len(rows)} strong">{dots}</div>'
                f'<div class="scount"><b>{n}</b>/{len(rows)} strong</div></div>')

    def count(key):
        rows = r["sets"].get(key) or []
        return sum(x["ratio"] >= STRONG for x in rows), len(rows)

    (a, n1), (b, n2) = count("new"), count("near_miss")
    head = (f'<p class="vline">{verdict_chip(r.get("verdict"))} <span>{a} of {n1} new texts were strong, against '
            f'{b} of {n2} near-miss twins (one-sided Fisher {esc(p_text(r.get("p_win")))}).</span></p>')
    strips = (strip("new", "s1", "New texts", "written to fire") + strip("near_miss", "s2", "Near-miss twins", "written not to")
              + strip("distant", "s6", "Distant texts", "far cases it still covers"))
    testers = r.get("testers") or []
    facts = []
    if testers:
        facts.append(("Each tester", "".join(f'<span class="tv {(t["verdict"] or "none").lower()}">{esc(t["tester"])}'
                                             f' {t["verdict"] or "–"}</span>' for t in testers)))
    if r.get("blind_ratio") is not None:
        facts.append(("Max attempt", f"<span><b>{percent(r['blind_ratio'])}</b> of the short-context max: the best of "
                                     f"the texts testers wrote to beat it (target 100%, "
                                     f"{'reached' if r['blind_pass'] else 'not reached'})</span>"))
    dl = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    return (f'<div class="plate verdict">{head}<div class="strips">{strips}</div>'
            f'<p class="key"><span><i class="on"></i>strong, at least {STRONG:.0%} of the short-context max</span>'
            f'<span><i></i>weaker</span></p><dl class="facts">{dl}</dl>{tester_table(testers) if testers else ""}</div>')


def tester_table(testers):
    cols = ("Tester", "Verdict", "New strong", "Near miss strong", "Distant strong", "p new &gt; near",
            "p near &gt; new", "Max attempt")
    head = "".join(f'<th scope="col">{h}</th>' for h in cols)
    rows = ""
    for t in testers:
        cells = (esc(t["tester"]), verdict_chip(t["verdict"]), percent(t["rate"]["new"]), percent(t["rate"]["near_miss"]),
                 percent(t["rate"]["distant"]), esc(pval(t.get("p_win"))), esc(pval(t.get("p_loss"))),
                 percent(t["blind_ratio"]))
        rows += "<tr>" + "".join(f'<td data-label="{h}">{c}</td>' for h, c in zip(cols, cells)) + "</tr>"
    return (f'<details class="testers"><summary>Each tester on their own</summary>'
            f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
            f'<p class="note">Share of each tester\'s texts that came out strong, and their one-sided Fisher '
            f'p-values.</p></details>')


def contrast_chart(pairs, hi):
    """A row per twin: the new text in full beside two bars, new and near miss, as % of the short-context max (an axis
    from 0 to hi), each value at its bar's end; gridlines every 50% and a dashed line at the strong threshold, drawn
    over the value labels so it never breaks. Bars and labels are HTML, so the text wraps and keeps its size."""
    ticks = range(0, int(round(100 * hi)) + 1, 50)

    def x(v):
        return f"{100 * min(max(v, 0), hi) / hi:.2f}"

    grid = '<div class="lines">' + "".join(f'<i class="gl" style="left:{x(t / 100)}%"></i>' for t in ticks) + "</div>"
    over = f'<div class="lines over"><i class="sl" style="left:{x(STRONG)}%"></i></div>'
    def bar(cls, t):
        """A bar and its value 3px past its end; a value that would run into the strong line from the left moves
        to just past the line instead (a CSS step: the push is 0 unless the gap to the line is under 33px)."""
        end, left = x(t["ratio"]), f"{x(t['ratio'])}% + 3px"
        if t["ratio"] < STRONG:
            gap = f"{float(x(STRONG)) - float(end):.2f}%"
            left += f" + min({gap} + 1px, max(0px, (33px - {gap}) * 1000))"
        tip = f' title="{esc(t["text"])}"' if cls == "b2" else ""
        return (f'<div class="brow"{tip}><i class="bar {cls}" style="width:{end}%"></i>'
                f'<span style="left:calc({left})">{100 * t["ratio"]:.0f}%</span></div>')

    labels = "".join(f'<span class="tk{" tk0" if t == 0 else ""}" style="left:{x(t / 100)}%">{t}%</span>' for t in ticks)
    rows = [f'<li class="crow head" aria-hidden="true"><div class="clab">New text</div><div class="plot">{grid}'
            f'<div class="area"><span class="stag" style="left:{x(STRONG)}%">strong, {STRONG:.0%}</span>{labels}</div>'
            f'{over}</div></li>']
    for new, near in pairs:
        who = f'<span class="who">{esc(new["tester"])}</span> ' if new.get("tester") else ""
        rows.append(
            f'<li class="crow"><div class="clab">{who}{esc(new["text"])}</div><div class="plot">{grid}<div class="area">'
            f'{bar("b1", new)}{bar("b2", near)}</div>{over}</div></li>')
    rows.append(f'<li class="crow axis" aria-hidden="true"><div class="clab"></div><div class="plot"><div class="area">'
                f'{labels}<span class="atitle">% of the short-context max</span></div></div></li>')
    label = (f"{len(pairs)} new texts against their near-miss twins as percent of the short-context max: "
             f"{sum(n['ratio'] >= STRONG for n, _ in pairs)} new and {sum(m['ratio'] >= STRONG for _, m in pairs)} "
             f"near misses strong")
    return (f'<figure class="plate chart"><ol class="crows" aria-label="{esc(label)}">{"".join(rows)}</ol>'
            f'<figcaption class="legend"><span class="k1">new text</span><span class="k2">near-miss twin</span>'
            f'<span class="kstrong">strong, at least {STRONG:.0%}</span></figcaption></figure>')


def level(share):
    """The shading step of a share of the short-context max: 0 (plain) to 4."""
    return sum(share >= b for b in BINS)


def ramp():
    steps = "".join(f'<span class="t h{i}">{esc(t)}</span>' for i, t in enumerate(BIN_LABELS, 1))
    return (f'<div class="ramp" aria-hidden="true"><span class="steps">{steps}</span>'
            f'<span class="rcap">of the short-context max</span>'
            f'<span class="pkey"><span class="t h3 pk">token</span> the peak</span></div>')


def example(kind, name, detail, x, pole, ref, full=None):
    """One text as tokens shaded by the pole's activation; plain text for scores saved without token activations.

    full is a corpus window's value in its full document, shown beside its value run alone."""
    if "tokens" in x:
        body, peak, tok = shaded(x["tokens"], x["acts"], pole, ref)
    else:
        body, peak, tok = esc(x["text"]), x["act"], None
    share = peak / ref if ref > 0 else 0.0
    if full is not None:
        detail = f'full document <b>{raw(full, pole)}</b> → run alone <b>{raw(peak, pole)}</b>'
        at = f' on “{esc(tok)}”' if tok else ""
    elif tok:
        at = f' on “{esc(tok)}”, {raw(peak, pole)}'
    else:
        at = f', {raw(peak, pole)}' if peak > 0 else ""
    if share < BINS[0]:
        at = ", no token reaches 5%"
    detail = f'<span class="detail">{detail}</span>' if detail else ""
    cut = '<span class="cut" title="cut from a longer document">…</span>' if kind == "corpus" else ""
    return (f'<div class="ex {kind}"><div class="ex-head"><span class="kind">{esc(name)}</span>{detail}'
            f'<span class="ex-v">peak <b>{100 * share:.0f}%</b>{at}</span></div><p class="toks">{cut}{body}</p></div>')


def shaded(tokens, acts, pole, ref):
    """(HTML, peak value, peak token): spans shaded in steps by the pole's activation as a share of ref, the peak
    underlined when it reaches the first step. Shaded tokens of one word join into one band, rounded at its ends."""
    s = 1 if pole == "+" else -1
    vals = [max(s * v, 0.0) for v in acts]
    peak = max(range(len(vals)), key=vals.__getitem__, default=None)
    lv = [level(v / ref) if ref > 0 else 0 for v in vals]
    marked = peak is not None and lv[peak] > 0
    on = [lv[i] > 0 or (marked and i == peak) for i in range(len(tokens))]
    out = []
    for i, (tok, v) in enumerate(zip(tokens, vals)):
        body = tok.lstrip(" ")
        out.append(esc(tok[:len(tok) - len(body)]))  # leading spaces stay unshaded
        if not body:
            continue
        text = esc(body).replace("\n", '<span class="nl">⏎</span>')
        if not on[i]:
            out.append(text)
            continue
        cls = ["t", f"h{lv[i]}"]
        if marked and i == peak:
            cls.append("pk")
        if i > 0 and on[i - 1] and tokens[i - 1].strip(" ") and not tok.startswith(" "):
            cls.append("jl")  # one band with the shaded token before it
        if i + 1 < len(tokens) and on[i + 1] and tokens[i + 1].strip(" ") and not tokens[i + 1].startswith(" "):
            cls.append("jr")
        out.append(f'<span class="{" ".join(cls)}">{text}</span>')
    if peak is None or vals[peak] <= 0:
        return "".join(out), 0.0, None
    return "".join(out), vals[peak], tokens[peak].strip().replace("\n", "⏎") or None


def card_details(card):
    if not card:
        return ""
    rows = "".join(f"<div><dt>{label}</dt><dd>{esc(card[k])}</dd></div>"
                   for k, label in CARD_LABELS.items() if card.get(k))
    rows += "".join(f"<div><dt>Anonymized {sign(p)} pole text</dt><dd>{esc(t)}</dd></div>"
                    for p, t in sorted(card.get("redacted", {}).items()))
    return (f'<details class="card"><summary>The card behind this page, as its author wrote it '
            f'<code>{esc(card["id"])}</code></summary><dl>{rows}</dl></details>')


# --- the index -------------------------------------------------------------------

def write_index(root):
    """Rewrite <root>/index.html from the metadata of every page under the root."""
    pages = []
    for path in root.glob("*/*.html"):
        if m := META.search(path.read_text()):
            pages.append({**json.loads(m[1]), "href": f"{path.parent.name}/{path.name}"})
    pages.sort(key=lambda p: (p["model"], [int(n) for n in re.findall(r"\d+", p["neuron"])]))
    groups = {}
    for p in pages:
        groups.setdefault(p["model"], []).append(p)
    body = []
    for model, ps in groups.items():
        body += [f'<section class="model"><h2>{esc(model)} <small>{len(ps)} neurons</small></h2><ul class="list">']
        for p in ps:
            poles = "".join(f'<span class="chip win">{sign(q)} WIN</span>' for q in p["poles"])
            body.append(
                f'<li><a href="{esc(p["href"])}"><span class="nid">{esc(p["neuron"])}</span>'
                f'<span class="ntitle">{esc(p["title"])}</span><span class="npoles">{poles}</span>'
                f'<span class="nscores"><span>clean {pips(p["clean"], "clean")}</span>'
                f'<span>interesting {pips(p["interesting"], "interesting")}</span></span>'
                f'<span class="nrun">run {esc(p["run"] or "–")} · '
                f'{time.strftime("%Y-%m-%d", time.localtime(p["time"]))}</span></a></li>')
        body.append("</ul></section>")
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(render(
        "index.html", count=len(pages), strong=f"{STRONG:.0%}",
        body="\n".join(body) or '<p class="note">No neuron has a winning pole yet.</p>'))
