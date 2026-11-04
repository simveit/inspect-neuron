"""Run-scoped library pages: a rebuild of one run must not delete another run's page."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers import MODEL, NEURON, paper_record, score_record, write_json

from nn import library
from nn.cli import library_update, winning_runs


class SaveScopeTest(unittest.TestCase):
    """library.save itself: ownership is read from the page's embedded meta JSON."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "neuron-library"
        self.page = self.root / "Qwen3-0.6B" / f"{NEURON}.html"
        self.r1 = paper_record(run="r1")
        self.r2 = paper_record(run="r2")

    def write_r2_page(self):
        page, action = library.save(self.root, MODEL, self.r2, None, {"+": score_record(run="r2")})
        self.assertEqual(action, "written")
        self.assertEqual(page, self.page)
        self.assertEqual(library.page_run(page), "r2")

    def test_win_writes_the_page_and_index(self):
        page, action = library.save(self.root, MODEL, self.r1, None, {"+": score_record(run="r1")})
        self.assertEqual(action, "written")
        self.assertEqual(library.page_run(page), "r1")
        self.assertIn(NEURON, (self.root / "index.html").read_text())

    def test_no_win_keeps_another_runs_page_when_it_still_wins(self):
        self.write_r2_page()
        before = self.page.read_text()
        page, action = library.save(self.root, MODEL, self.r1, None, {}, other_wins=True)
        self.assertEqual(action, "kept")
        self.assertEqual(page.read_text(), before)
        self.assertEqual(library.page_run(page), "r2")
        self.assertIn(NEURON, (self.root / "index.html").read_text())

    def test_no_win_removes_the_same_runs_page_even_when_another_run_wins(self):
        self.write_r2_page()
        self.assertEqual(library.page_run(self.page), "r2")
        page, action = library.save(self.root, MODEL, self.r2, None, {}, other_wins=True)
        self.assertEqual(action, "removed")
        self.assertFalse(page.exists())

    def test_no_win_removes_a_stale_other_run_page_when_nothing_wins_anymore(self):
        self.write_r2_page()
        page, action = library.save(self.root, MODEL, self.r1, None, {}, other_wins=False)
        self.assertEqual(action, "removed")
        self.assertFalse(page.exists())

    def test_no_win_with_no_page_is_absent(self):
        page, action = library.save(self.root, MODEL, self.r1, None, {}, other_wins=False)
        self.assertEqual(action, "absent")
        self.assertFalse(page.exists())
        self.assertTrue((self.root / "index.html").exists())

    def test_a_file_without_page_meta_is_never_removed(self):
        self.page.parent.mkdir(parents=True)
        self.page.write_text("<html>handmade page, not ours</html>")
        page, action = library.save(self.root, MODEL, self.r1, None, {}, other_wins=False)
        self.assertEqual(action, "kept")
        self.assertEqual(page.read_text(), "<html>handmade page, not ours</html>")


class CliScopeTest(unittest.TestCase):
    """winning_runs and library_update: the store decides which other run still wins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.store = SimpleNamespace(path=base / "store", meta={"model": MODEL})
        (self.store.path / "scores").mkdir(parents=True)
        self.library = base / "neuron-library"
        self.page = self.library / "Qwen3-0.6B" / f"{NEURON}.html"
        self.args = SimpleNamespace(library=str(self.library), run="r1")

    def add_score(self, record):
        write_json(self.store.path / "scores" / f"{record['run']}_{record['pole']}_{record['time']}.json", record)

    def update(self, run, paper_run=None):
        self.args.run = run
        return library_update(self.args, self.store, paper_record(run=paper_run or run))

    def test_winning_runs_uses_the_newest_score_per_run_and_pole(self):
        self.add_score(score_record(run="r1", time=1.0))
        self.add_score(score_record(run="r2", time=2.0))
        self.assertEqual(winning_runs(self.store, NEURON), {"r1", "r2"})
        self.add_score(score_record(run="r1", time=3.0, verdict="TIE"))
        self.assertEqual(winning_runs(self.store, NEURON), {"r2"})
        self.add_score(score_record(run="r2", pole="-", time=4.0, verdict="LOSS"))
        self.assertEqual(winning_runs(self.store, NEURON), {"r2"})
        self.assertEqual(winning_runs(self.store, "L9.N9"), set())

    def test_rebuild_of_an_old_run_keeps_the_newer_runs_page(self):
        self.add_score(score_record(run="r2", time=2.0))
        message = self.update("r2")
        self.assertIn("library page", message)
        self.assertEqual(library.page_run(self.page), "r2")
        before = self.page.read_text()

        message = self.update("r1")  # r1 has no scores at all
        self.assertIn("kept the library page of run r2", message)
        self.assertEqual(self.page.read_text(), before)

    def test_rebuild_removes_the_runs_own_page_when_its_newest_score_is_no_win(self):
        self.add_score(score_record(run="r1", time=1.0))
        self.update("r1")
        self.assertEqual(library.page_run(self.page), "r1")

        self.add_score(score_record(run="r1", time=2.0, verdict="TIE"))
        self.add_score(score_record(run="r2", time=3.0))
        message = self.update("r1")
        self.assertIn("removed library page", message)
        self.assertFalse(self.page.exists())

    def test_rebuild_removes_a_stale_page_when_no_run_wins_anymore(self):
        self.add_score(score_record(run="r2", time=1.0))
        self.update("r2")
        self.assertTrue(self.page.exists())
        self.add_score(score_record(run="r2", time=2.0, verdict="LOSS"))
        message = self.update("r1")
        self.assertIn("removed library page", message)
        self.assertFalse(self.page.exists())


if __name__ == "__main__":
    unittest.main()
