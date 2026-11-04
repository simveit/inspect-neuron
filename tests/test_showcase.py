"""Showcase records: Markdown escaping, dose/judge rendering and strict record validation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers import MODEL

from nn.showcase import (STEER_LIMITS, check_classification, check_control, check_record,
                         dose_md, judge_md, md_fence, md_inline, md_quote, period,
                         showcase_md, steer_int, steer_number, steer_text)


def valid_payload(**over):
    """A payload that check_record accepts, before overrides."""
    payload = {
        "neuron": "L8.N2373", "run": "r1", "model": MODEL,
        "query": "What should I do?",
        "mode": "add", "position": "all", "strength": 128.0,
        "unit": 0.25, "activation": 32.0, "temperature": 0.8, "seed": 0, "chat": True,
        "base": "BASE text", "steered": "STEERED text", "control": None,
        "classification": {"coherent": True, "on_concept": True, "unaligned": False,
                           "movement": 2, "note": ""},
        "selection": "why this run", "recipe": "how it was made", "reproduce": "nn generate ...",
    }
    payload.update(over)
    return payload


class FakeStore:
    allowed = {(8, 2373), (2, 5)}

    def __init__(self):
        self.meta = {"model": MODEL}

    def check(self, layer, idx):
        if (layer, idx) not in self.allowed:
            raise ValueError(f"neuron L{layer}.N{idx} out of range")


class MarkdownTest(unittest.TestCase):
    def test_inline_folds_newlines_and_escapes_markdown(self):
        self.assertEqual(md_inline("a\nb"), "a b")
        self.assertEqual(md_inline("a\r\nb\x00c"), "a bc")
        escaped = md_inline('*x* <y> & "z" #tag [q](r)')
        for ch in "*<>&\"#[]()":
            self.assertIn("\\" + ch, escaped)

    def test_quote_marks_every_line(self):
        self.assertEqual(md_quote("a\n\nb"), "> a\n>\n> b")
        self.assertEqual(md_quote("x\r\ny"), "> x\n> y")
        self.assertIn("\\*", md_quote("*x*"))

    def test_fence_outruns_its_own_backticks(self):
        self.assertEqual(md_fence("nn top L1.N2"), "```bash\nnn top L1.N2\n```")
        fenced = md_fence("echo ``` inner")
        self.assertTrue(fenced.startswith("````bash\n"))
        self.assertTrue(fenced.endswith("\n````"))

    def test_period(self):
        self.assertEqual(period("done"), "done.")
        self.assertEqual(period("done?"), "done?")
        self.assertEqual(period(""), "")
        self.assertEqual(period("done. "), "done.")

    def test_dose_always_names_its_unit(self):
        self.assertEqual(dose_md({"mode": "add", "strength": 128.0, "position": "all", "ref": "sigma"}),
                         "add +128σ at positions all")
        self.assertEqual(dose_md({"mode": "set", "strength": -8.0, "position": "new", "ref": "sigma"}),
                         "set -8σ at positions new")
        self.assertEqual(dose_md({"mode": "scale", "strength": 2.0, "position": "all", "ref": "sigma"}),
                         "scale ×2 at positions all")
        detail = dose_md({"mode": "add", "strength": 128.0, "position": "all", "ref": "sigma",
                          "unit": 0.25, "activation": 32.0}, detail=True)
        self.assertIn("(unit 0.2500, ≈ +32.00 activation)", detail)
        resid = dose_md({"mode": "set", "strength": 0.3, "position": "new", "ref": "resid",
                         "ref_unit": 0.5, "activation": 1.5}, detail=True)
        self.assertIn("set +0.3 resid at positions new", resid)
        self.assertIn("(1 resid = 0.5000 activation, ≈ +1.50 activation)", resid)

    def test_judge_md_copies_the_verdict_unchanged(self):
        coherent = {"coherent": True, "on_concept": True, "unaligned": False, "movement": 2, "note": ""}
        self.assertEqual(judge_md({"classification": coherent}),
                         "Judge: coherent **yes**, on-concept **yes**, unaligned **no**, movement **2/3**.")
        noted = judge_md({"classification": {**coherent, "note": "a note"}})
        self.assertTrue(noted.endswith("a note."))
        question = judge_md({"classification": {**coherent, "note": "really?"}})
        self.assertTrue(question.endswith("really?"))
        incoherent = judge_md({"classification": {**coherent, "coherent": False}})
        self.assertIn("shown for its concept shift, not as a usable answer", incoherent)
        unaligned = judge_md({"classification": {**coherent, "unaligned": True}})
        self.assertIn("The judge marked it unaligned.", unaligned)

    def test_showcase_section_renders_records_in_neuron_order(self):
        a = check_record(type("A", (), {"run": "r1"})(), FakeStore(), valid_payload())
        b = check_record(type("A", (), {"run": "r1"})(), FakeStore(),
                         valid_payload(neuron="L2.N5"))
        out = "\n".join(showcase_md({"L8.N2373": a, "L2.N5": b}, "r1"))
        self.assertIn("## Showcase", out)
        self.assertIn("Run r1.", out)
        self.assertLess(out.index("### L2.N5"), out.index("### L8.N2373"))
        self.assertIn("**Query.** What should I do?", out)
        self.assertIn("**BASE.**", out)
        self.assertIn("> BASE text", out)
        self.assertIn("**STEERED, add +128σ at positions all (unit 0.2500, ≈ +32.00 activation).**", out)
        self.assertIn("**CONTROL.** None recorded.", out)
        self.assertIn("**Reproduce.**", out)
        self.assertIn("```bash\nnn generate ...\n```", out)

    def test_showcase_section_escapes_untrusted_text(self):
        rec = check_record(type("A", (), {"run": "r1"})(), FakeStore(),
                           valid_payload(query="<script>alert(1)</script>",
                                         base="line *one*\nline [two]"))
        out = "\n".join(showcase_md({"L8.N2373": rec}))
        self.assertNotIn("<script>", out)
        self.assertIn("\\<script\\>", out)
        self.assertIn("> line \\*one\\*", out)
        self.assertIn("> line \\[two\\]", out)


class ValidatorTest(unittest.TestCase):
    def test_numbers_reject_bool_strings_and_non_finite(self):
        self.assertEqual(steer_number(2, "x"), 2.0)
        for bad in (True, "2", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                steer_number(bad, "x")

    def test_int_ranges(self):
        self.assertEqual(steer_int(2.0, "x", 0, 3), 2)
        for bad in (2.5, -1, 4, True):
            with self.assertRaises(ValueError):
                steer_int(bad, "x", 0, 3)

    def test_text_strips_carriage_returns_and_nul(self):
        self.assertEqual(steer_text("a\r\nb\x00c", "x", 10), "a\nbc")
        for bad in (None, 2):
            with self.assertRaises(ValueError):
                steer_text(bad, "x", 10)
        with self.assertRaises(ValueError):
            steer_text("   ", "x", 10)
        self.assertEqual(steer_text("   ", "x", 10, allow_empty=True), "   ")
        with self.assertRaises(ValueError):
            steer_text("abcdef", "x", 5)

    def test_control(self):
        control = check_control({"mode": "set", "position": "new", "strength": -8, "text": "ctrl"})
        self.assertEqual(control, {"mode": "set", "position": "new", "strength": -8.0,
                                   "text": "ctrl", "ref": "sigma"})
        self.assertIsNone(check_control(None))
        for bad in ("x", {"mode": "add", "position": "all", "strength": 1, "text": "t", "extra": 1},
                    {"mode": "add", "position": "all", "strength": 1},
                    {"mode": "nope", "position": "all", "strength": 1, "text": "t"},
                    {"mode": "add", "position": "nope", "strength": 1, "text": "t"},
                    {"mode": "add", "position": "all", "strength": 1, "text": "t", "ref": "nope"},
                    {"mode": "add", "position": "all", "strength": 1, "text": "t", "ref_unit": -1}):
            with self.assertRaises(ValueError):
                check_control(bad)

    def test_classification(self):
        good = {"coherent": True, "on_concept": False, "unaligned": False, "movement": 2, "note": ""}
        self.assertEqual(check_classification(good),
                         {"coherent": True, "on_concept": False, "unaligned": False,
                          "movement": 2, "note": ""})
        for bad in ({**good, "coherent": 1}, {**good, "movement": 4}, {**good, "movement": True},
                    {**good, "note": "x" * (STEER_LIMITS["note"] + 1)},
                    {k: v for k, v in good.items() if k != "note"},
                    {**good, "extra": 1}):
            with self.assertRaises(ValueError):
                check_classification(bad)

    def test_check_record_normalizes_and_fills_defaults(self):
        a = type("A", (), {"run": "r1"})()
        rec = check_record(a, FakeStore(), valid_payload())
        self.assertEqual(rec["run"], "r1")
        self.assertEqual(rec["model"], MODEL)
        self.assertEqual(rec["ref"], "sigma")
        self.assertEqual(rec["seed"], 0)
        self.assertIsInstance(rec["time"], float)
        self.assertNotIn("ref_unit", rec)

    def test_check_record_rejects_unknown_missing_and_mismatched_fields(self):
        a = type("A", (), {"run": "r1"})()
        cases = [
            valid_payload(extra=1),
            {k: v for k, v in valid_payload().items() if k != "query"},
            valid_payload(neuron="nope"),
            valid_payload(run="r2"),
            valid_payload(run="bad run"),
            valid_payload(mode="nope"),
            valid_payload(position="nope"),
            valid_payload(ref="resid"),  # ref_unit required
            valid_payload(chat="yes"),
            valid_payload(temperature=-1),
            valid_payload(unit=-1),
            valid_payload(seed=1.5),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    check_record(a, FakeStore(), payload)

    def test_check_record_accepts_resid_with_ref_unit(self):
        a = type("A", (), {"run": "r1"})()
        rec = check_record(a, FakeStore(), valid_payload(ref="resid", ref_unit=0.5))
        self.assertEqual(rec["ref"], "resid")
        self.assertEqual(rec["ref_unit"], 0.5)

    def test_check_record_requires_a_run_label(self):
        a = type("A", (), {"run": None})()
        payload = valid_payload()
        payload.pop("run")
        with self.assertRaises(ValueError):
            check_record(a, FakeStore(), payload)


if __name__ == "__main__":
    unittest.main()
