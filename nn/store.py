"""Read side of a harvest store; numpy and tokenizers only, so the CLI starts fast.

A store is one directory (nn.harvest has the layout). Two layouts share it:

  full      meta.json has no "neurons" key; vals.npy is [L, I, 4, k] and
            windows.npy is [L, I, 4, k, W], one row for every neuron of the model.
  targeted  meta.json has "neurons": [[layer, idx], ...] with S entries; row r of
            vals.npy is [4, k] and row r of windows.npy is [4, k, W], for those S
            neurons only. fires.npy stays dense [L, I, 2] in both layouts, so
            alive() and select keep working (untracked neurons have zero counts).

Only the tracked neurons of a targeted store can be read; check() says which
store it is when a neuron is missing.
"""

import json
import re
from pathlib import Path

import numpy as np

REGIMES = ["+x+", "-x-", "+x-", "-x+"]  # gate sign x up sign; first two are the + pole
POLE_REGIMES = {"+": (0, 1), "-": (2, 3)}


def parse_neuron(s):
    m = re.fullmatch(r"L(\d+)\.N(\d+)", s)
    if not m:
        raise ValueError(f"bad neuron id {s!r}, expected e.g. L12.N3456")
    return int(m[1]), int(m[2])


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.neurons = self.meta.get("neurons")  # [[layer, idx], ...] for a targeted store, else None
        self.vals = np.load(self.path / "vals.npy", mmap_mode="r")
        self.windows = np.load(self.path / "windows.npy", mmap_mode="r")
        self.fires = np.load(self.path / "fires.npy", mmap_mode="r")
        self._rows = None
        if self.neurons is not None:
            self._rows = {(int(l), int(i)): r for r, (l, i) in enumerate(self.neurons)}
            if len(self._rows) != len(self.neurons):
                raise ValueError(f"{self.path} lists duplicate neurons in meta.json")
            if self.vals.shape[0] != len(self.neurons) or self.windows.shape[0] != len(self.neurons):
                raise ValueError(f"{self.path} is a targeted store with {len(self.neurons)} neurons in meta.json "
                                 f"but vals.npy {self.vals.shape} / windows.npy {self.windows.shape}")
        self._tok = None

    @property
    def shape(self):
        """(layers, width) of the model; a targeted store's arrays carry S rows instead."""
        if "layers" in self.meta and "width" in self.meta:
            return int(self.meta["layers"]), int(self.meta["width"])
        if self.neurons is None:
            return self.vals.shape[:2]
        raise ValueError(f"{self.path}/meta.json lacks layers/width for a targeted store")

    def _missing(self, layer, idx):
        return ValueError(f"{self.path} is a targeted store of {len(self.neurons)} neurons "
                          f"and does not contain L{layer}.N{idx}")

    def _row(self, layer, idx):
        """Index tuple into vals/windows: (layer, idx) in a full store, (row,) in a targeted one."""
        if self.neurons is None:
            return layer, idx
        row = self._rows.get((layer, idx))
        if row is None:
            raise self._missing(layer, idx)
        return (row,)

    def check(self, layer, idx):
        L, width = self.shape
        if not (0 <= layer < L and 0 <= idx < width):
            raise ValueError(f"neuron L{layer}.N{idx} out of range: {L} layers x {width} neurons")
        if self.neurons is not None and (layer, idx) not in self._rows:
            raise self._missing(layer, idx)

    def corpus_max(self, layer, idx):
        """Largest magnitude seen in each pole: {"+": max h, "-": max -h} (0 if never)."""
        v = self.vals[self._row(layer, idx)][:, 0]
        return {p: max(float(v[list(rs)].max()), 0.0) for p, rs in POLE_REGIMES.items()}

    def examples(self, layer, idx, regimes):
        """[(value, regime index, window ids)] for the given regimes, strongest first."""
        r = self._row(layer, idx)
        v, w = self.vals[r], self.windows[r]
        rows = [(float(x), g, w[g, j]) for g in regimes for j, x in enumerate(v[g]) if x > 0]
        return sorted(rows, key=lambda row: -row[0])

    def alive(self, dead_eps, min_fires):
        """[L, I] mask: some pole reached dead_eps and the neuron fired min_fires times."""
        if self.neurons is None:
            peak = np.asarray(self.vals[:, :, :, 0]).max(-1)
        else:
            peak = np.zeros(self.shape, dtype=np.float32)  # untracked neurons stay at zero
            for r, (l, i) in enumerate(self.neurons):
                peak[int(l), int(i)] = np.asarray(self.vals[r, :, 0]).max()
        return (peak >= dead_eps) & (np.asarray(self.fires).sum(-1) >= min_fires)

    @property
    def tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        return self._tok

    def decode(self, ids):
        ids = [int(i) for i in ids if i >= 0]
        return self.tokenizer.decode(ids, skip_special_tokens=False) if ids else ""

    def window_text(self, ids):
        """Window as text with the peak token in [[ ]] and newlines shown as a return sign."""
        b = self.meta["before"]
        text = self.decode(ids[:b]) + "[[" + self.decode(ids[b:b + 1]) + "]]" + self.decode(ids[b + 1:])
        return text.replace("\n", "⏎")
