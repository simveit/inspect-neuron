"""Guards that the repository never leaks machine-specific or internal-only strings."""

import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Skip operational dirs and tests/.
SKIP_DIRS = {".git", "reports", "runs", "run", "neuron-library", "__pycache__",
             ".venv", "build", "dist", "tests"}
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npy", ".npz",
                 ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".arrow", ".parquet"}
# SHA-256 of the private tokens (host aliases, user names, internal script and folder names),
# so this guard does not ship the strings it looks for.
BLOCKED = {
    "515f3f510efacd0b0a709746b30f5674b298a6fbed90b150d930a11755031b58",
    "0a5d17d3b19f82f8340d3977609aa9e86b4ad8b9bd71bd9eced9271f1d5b2e4a",
    "043af9454f21ddf0a4d42b9893470a4243f31b2d9139d10d4de0d68a7526578f",
    "b8c5512fcbad792a4fd9755283f3fd75254617fc522ffb5355ae1cc351f50692",
    "fb6c952d2b3e531ad0f985e3f7ba01d483054ddab0b350c87da3ff22715533b1",
    "7769142444e7e8c796097f6abff89c2ecbd6183fbcf37292b6ad18a88a71b58f",
    "1762fb3aabdc1ea7b6aae00ab898e54ad4f614e6b4dccedbfd0f7e863e36701d",
    "d3c418e9280d97aa0d71023fe7dc154eb396416f5552c5dc2cb47029fc9e4f92",
}
TOKEN = re.compile(r"[A-Za-z0-9_.-]+")


def text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def candidates(text):
    """Every token and every hyphen-joined run of its parts (`a-b-c` -> `a`, `b`, `a-b`, ...)."""
    for token in set(TOKEN.findall(text)):
        parts = token.split("-")
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                yield "-".join(parts[i:j])


class NoMachineLeaks(unittest.TestCase):
    def test_no_machine_or_internal_strings(self):
        hits = []
        for path in text_files():
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for word in set(candidates(text)):
                if hashlib.sha256(word.encode()).hexdigest() in BLOCKED:
                    hits.append(f"{path.relative_to(ROOT)}: {word}")
        self.assertFalse(hits, "machine/internal strings in the tree: " + "; ".join(hits[:10]))

    def test_ignored_operational_directories(self):
        ignore = (ROOT / ".gitignore").read_text()
        for needed in ("run/", "reports/", "runs/", "data/", "models/"):
            self.assertIn(needed, ignore)


if __name__ == "__main__":
    unittest.main()
