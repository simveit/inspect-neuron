"""Pure helpers in nn.cli that need no model, server, store or torch."""

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nn.cli import (ALPHA, STRONG, cell, check_length, contrast, fisher, float_list, judge, pct,
                    percent, pole_value, pval, summary_line, strong_text, twins, verdict_text)
from nn.cli import show as show_text


def result(new_strong=5, near_strong=0, n_new=5, n_near=5, blind=0.4):
    """A scored pole: the first `new_strong` new rows and `near_strong` near misses are strong."""
    def row(strong, text):
        ratio = 0.5 if strong else 0.1
        return {"text": text, "act": ratio, "ratio": ratio, "corpus_ratio": ratio,
                "tokens": [text], "acts": [ratio]}

    return {"neuron": "L1.N2", "pole": "+", "run": "r1", "card": None, "tester": None,
            "corpus_max": 1.0, "short_max": 1.0,
            "sets": {"new": [row(i < new_strong, f"new {i}") for i in range(n_new)],
                     "near_miss": [row(i < near_strong, f"near {i}") for i in range(n_near)],
                     "distant": [], "blind_max": [{"act": blind}]}}


class FormattingTest(unittest.TestCase):
    def test_pct_percent_pval(self):
        self.assertEqual(pct(1, 4), "25%")
        self.assertEqual(pct(1, 0), "n/a")
        self.assertEqual(percent(None), "-")
        self.assertEqual(percent(0.256), "26%")
        self.assertEqual(pval(None), "-")
        self.assertEqual(pval(0.0004), "<0.001")
        self.assertEqual(pval(0.5), "0.500")

    def test_cell_escapes_and_cuts(self):
        self.assertEqual(cell("a|b\nc", 10), "a\\|b c")
        self.assertEqual(cell("abcdef", 4), "abc…")
        self.assertEqual(cell(None), "")

    def test_show_marks_newlines(self):
        text = show_text("a\nb")
        self.assertIn("⏎", text)
        self.assertTrue(text.startswith("'"))

    def test_float_list(self):
        self.assertEqual(float_list(""), [])
        self.assertEqual(float_list("1, -2.5"), [1.0, -2.5])
        for bad in ("1,,2", "x", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                float_list(bad)

    def test_check_length(self):
        check_length({"a": "12345"}, 5)
        with self.assertRaises(ValueError) as e:
            check_length({"a": "123456"}, 5)
        self.assertIn("over 5 characters", str(e.exception))


class ScoreMathTest(unittest.TestCase):
    def test_pole_value_skips_the_separator(self):
        acts = [9.0, -0.6, 0.4]
        self.assertAlmostEqual(pole_value(acts, "+"), 0.4)
        self.assertAlmostEqual(pole_value(acts, "-"), 0.6)
        self.assertEqual(pole_value([], "+"), 0.0)
        self.assertEqual(pole_value([9.0, -1.0], "+"), 0.0)

    def test_fisher_one_sided(self):
        self.assertAlmostEqual(fisher(5, 5, 0, 5), 1 / 252)
        self.assertAlmostEqual(fisher(0, 5, 5, 5), 1.0)
        self.assertAlmostEqual(fisher(0, 5, 0, 5), 1.0)
        self.assertAlmostEqual(fisher(3, 5, 0, 5), 21 / 252)

    def test_judge_win_tie_loss(self):
        win = judge(result())
        self.assertEqual(win["verdict"], "WIN")
        self.assertLess(win["p_win"], ALPHA)
        self.assertEqual(win["rate"]["new"], 1.0)
        self.assertEqual(win["rate"]["distant"], None)
        self.assertEqual(win["blind_ratio"], 0.4)
        self.assertFalse(win["blind_pass"])

        tie = judge(result(new_strong=2, near_strong=2))
        self.assertEqual(tie["verdict"], "TIE")

        loss = judge(result(new_strong=0, near_strong=5))
        self.assertEqual(loss["verdict"], "LOSS")
        self.assertLess(loss["p_loss"], ALPHA)

    def test_judge_without_both_sets_has_no_verdict(self):
        res = {"neuron": "L1.N2", "pole": "+", "short_max": 1.0,
               "sets": {"new": [], "near_miss": [], "distant": [], "blind_max": []}}
        out = judge(res)
        self.assertIsNone(out["verdict"])
        self.assertIsNone(out["p_win"])
        self.assertIsNone(out["blind_ratio"])

    def test_threshold_is_the_shared_strong_constant(self):
        res = result(new_strong=1, near_strong=0, n_new=2)
        res["sets"]["new"][0]["ratio"] = STRONG
        res["sets"]["new"][1]["ratio"] = STRONG - 0.001
        self.assertEqual(judge(res)["rate"]["new"], 0.5)

    def test_verdict_and_strong_text(self):
        self.assertEqual(verdict_text({"verdict": "WIN", "testers": []}), "WIN")
        pooled = {"verdict": "WIN", "testers": [{"verdict": "WIN"}, {"verdict": "TIE"}]}
        self.assertEqual(verdict_text(pooled), "WIN (WIN, TIE)")
        strong = strong_text(judge(result(new_strong=2, near_strong=0)))
        self.assertTrue(strong.startswith("Strong: new 2/5, near miss 0/5, distant -."))
        self.assertIn("p = 0.222", strong)

    def test_summary_line(self):
        res = judge(result())
        res["testers"] = [{"verdict": "WIN"}, {"verdict": "TIE"}]
        line = summary_line(res)
        self.assertIn("L1.N2 + pooled over 2 testers", line)
        self.assertIn("short-context max 1.000 (corpus 1.000)", line)
        self.assertIn("blind max 40% fail", line)
        self.assertIn("WIN (WIN, TIE)", line)

    def test_twins_pairs_testers_and_widest_gap_first(self):
        res = {"sets": {
            "new": [{"text": "wide", "ratio": 0.9, "act": 0.9, "tester": "t1"},
                    {"text": "narrow", "ratio": 0.4, "act": 0.4, "tester": "t2"}],
            "near_miss": [{"text": "wide near", "ratio": 0.1, "act": 0.1, "tester": "t1"},
                          {"text": "narrow near", "ratio": 0.35, "act": 0.35, "tester": "t2"}],
        }}
        pairs = twins(res)
        self.assertEqual([p[0]["text"] for p in pairs], ["wide", "narrow"])
        rows = contrast(res)
        self.assertEqual(rows[0]["new"], "wide")
        self.assertEqual(rows[0]["near_miss"], "wide near")
        self.assertAlmostEqual(rows[0]["new_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
