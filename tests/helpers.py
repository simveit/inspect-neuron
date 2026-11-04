"""Shared fixtures for the tests: the repo root and minimal, renderable store records.

The tests import numpy (a declared dependency) but never torch, transformers or datasets, so
`python3 -m unittest discover -s tests -v` runs on any machine.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "Qwen/Qwen3-0.6B"
NEURON = "L1.N2"


def score_record(neuron=NEURON, pole="+", run="r1", verdict="WIN", time=1.0):
    """A minimal score record that nn.library can render: the newest score of one pole."""
    return {
        "neuron": neuron, "pole": pole, "run": run, "time": time, "verdict": verdict,
        "card": None, "tester": None, "corpus_max": 1.0, "short_max": 1.0,
        "blind_ratio": None, "blind_pass": None, "p_win": 0.01, "p_loss": 1.0,
        "sets": {
            "new": [{"text": "a new text", "act": 0.5, "ratio": 0.5, "corpus_ratio": 0.5,
                     "tokens": ["a", " new", " text"], "acts": [0.5, 0.1, 0.0]}],
            "near_miss": [{"text": "a near miss", "act": 0.1, "ratio": 0.1, "corpus_ratio": 0.1,
                           "tokens": ["a", " near", " miss"], "acts": [0.1, 0.0, 0.0]}],
            "distant": [],
        },
        "top": [], "testers": [],
    }


def paper_record(neuron=NEURON, run="r1", time=1.0):
    """A minimal paper record, enough for nn.library to write a page."""
    return {
        "neuron": neuron, "run": run, "card": None, "title": "A test neuron",
        "summary": "It does a thing.", "clean": 4, "clean_why": "One concept.",
        "interesting": 3, "interesting_why": "Mostly explained.", "time": time, "contrast": {},
    }


def write_json(path, obj):
    """Write obj as JSON under path, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path
