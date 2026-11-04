"""Docs and the DSH workflow stay consistent, portable and free of removed options."""

import json
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers import ROOT

WORKFLOW = ROOT / ".dsh" / "workflows" / "blind-neuron-test.js"
META = ROOT / ".dsh" / "workflows" / "blind-neuron-test.meta.json"
DOCS = ("README.md", "AGENTS.md", "DESIGN_DOC.md")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

def project_files():
    """The tracked, non-article, non-artifact files this test guards."""
    files = [ROOT / name for name in ("README.md", "AGENTS.md", "DESIGN_DOC.md", "pyproject.toml",
                                      ".gitignore", WORKFLOW.relative_to(ROOT), META.relative_to(ROOT))]
    files += sorted((ROOT / "nn").rglob("*.py"))
    files += sorted((ROOT / "nn" / "templates").glob("*"))
    files += sorted((ROOT / "scripts").glob("*"))
    files += sorted((ROOT / "run.example").glob("*"))
    return [p for p in files if p.is_file()]


class DocsTest(unittest.TestCase):
    def test_relative_markdown_links_resolve(self):
        for name in DOCS:
            text = (ROOT / name).read_text()
            for target in LINK.findall(text):
                target = target.strip().strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                with self.subTest(doc=name, target=target):
                    self.assertTrue((ROOT / target).exists(), f"{name} links a missing {target}")

    def test_readme_quickstart_points_at_the_runner_and_templates(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn("scripts/nn.sh", readme)
        self.assertIn("run.example", readme)

    def test_runner_and_run_templates_exist(self):
        runner = ROOT / "scripts" / "nn.sh"
        self.assertTrue(runner.is_file())
        self.assertTrue(os.access(runner, os.X_OK), "scripts/nn.sh must be executable")
        for name in ("README.md", "remote.sh.example", "run.env.example"):
            self.assertTrue((ROOT / "run.example" / name).is_file(), f"run.example/{name} is missing")


class WorkflowTest(unittest.TestCase):
    def test_meta_parses_and_has_the_ten_phases(self):
        meta = json.loads(META.read_text())
        self.assertEqual(meta["name"], "blind-neuron-test")
        titles = [p["title"] for p in meta["phases"]]
        self.assertEqual(titles, ["Screen", "Setup", "Inspect", "Synthesize", "Anonymize", "Test",
                                  "Score", "Paper", "Steer", "Report"])

    def test_phase_calls_and_agent_phases_match_the_meta(self):
        body = WORKFLOW.read_text()
        titles = [p["title"] for p in json.loads(META.read_text())["phases"]]
        called = re.findall(r"\bphase\('([^']+)'\)", body)
        self.assertTrue(called)
        for title in called:
            self.assertIn(title, titles)
        self.assertEqual(called, sorted(called, key=titles.index))
        agent_phases = list(dict.fromkeys(re.findall(r"\bopts\([^\n]*?,\s*'([^']+)'", body)))
        self.assertEqual(agent_phases, titles)

    def test_workflow_uses_a_configurable_local_runner(self):
        body = WORKFLOW.read_text()
        self.assertIn("runner: './scripts/nn.sh'", body)
        self.assertNotIn("NN_STORE=${cfg.store}", body)
        self.assertNotIn("./" + "remote/" + "nn" + "-remote.sh", body)
        self.assertIn("NN_STORE=${sh(cfg.store)}", body)
        self.assertIn("NN_SERVER=${sh(cfg.server)}", body)
        self.assertIn("const sh = v =>", body)
        self.assertIn("NN_RUN=${cfg.run}", body)
        self.assertNotIn("remote/" + "nn" + "-remote.sh", META.read_text())


if __name__ == "__main__":
    unittest.main()
