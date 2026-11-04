"""Text corpora for nn.rank and nn.harvest.

By default both stream FineWeb (sample-10BT). A run can instead read an
arbitrary Hugging Face dataset (`--dataset`, with `--config`, `--split`,
`--field`, `--where`) or a JSONL / plain-text file (`--texts`), capped with
`--max-docs`. The source spec is stored in meta.json, so a resume cannot
silently switch corpora.

A document is one dataset row or one file line: the chosen `--field` columns
joined with newlines. `--where KEY=VALUE` keeps only rows whose column equals
the value (case-insensitive, useful for e.g. is_safe=false). Files are
resumable by line position, datasets by the streaming state.
"""

import json
from pathlib import Path

FINE_WEB = {"kind": "fineweb", "fields": ["text"], "where": {}}


def make_source(dataset=None, config=None, split=None, fields=None, where=None, texts=None):
    """The source spec for --dataset/--config/--split/--field/--where/--texts."""
    if dataset and texts:
        raise ValueError("pass either --dataset or --texts, not both")
    if dataset:
        return {"kind": "dataset", "id": dataset, "config": config, "split": split or "train",
                "fields": fields or ["text"], "where": where or {}}
    if texts:
        path = Path(texts).expanduser()
        if not path.exists():
            raise ValueError(f"no text file {texts}")
        return {"kind": "texts", "path": str(path), "fields": fields or ["text"], "where": where or {}}
    if config or split or fields or where:
        raise ValueError("--config, --split, --field and --where need --dataset or --texts")
    return dict(FINE_WEB)


def normalize(spec):
    """Canonical source spec for a source dict, filling defaults and checking the kind."""
    if not isinstance(spec, dict) or "kind" not in spec:
        raise ValueError(f"bad text source {spec!r}")
    kind = spec["kind"]
    if kind == "fineweb":
        return dict(FINE_WEB)
    if kind == "dataset":
        if not spec.get("id"):
            raise ValueError("a dataset source needs an id")
        return {"kind": "dataset", "id": spec["id"], "config": spec.get("config"),
                "split": spec.get("split") or "train",
                "fields": spec.get("fields") or ["text"], "where": spec.get("where") or {}}
    if kind == "texts":
        if not spec.get("path"):
            raise ValueError("a texts source needs a path")
        return {"kind": "texts", "path": spec["path"],
                "fields": spec.get("fields") or ["text"], "where": spec.get("where") or {}}
    raise ValueError(f"unknown text source kind {kind!r}")


def describe(spec):
    """Short human-readable source, for logs and error messages."""
    kind = spec["kind"]
    if kind == "fineweb":
        return "FineWeb sample-10BT"
    if kind == "dataset":
        parts = [spec["id"], f"split {spec['split']}"]
        if spec["config"]:
            parts.insert(1, f"config {spec['config']}")
        if spec["fields"] != ["text"]:
            parts.append("field " + ",".join(spec["fields"]))
        if spec["where"]:
            parts.append("where " + ",".join(f"{k}={v}" for k, v in spec["where"].items()))
        return ", ".join(parts)
    return f"{spec['path']} (field {','.join(spec['fields'])})"


class Texts:
    """Iterate the documents of one source, with a resumable state."""

    def __init__(self, spec, shard=(0, 1), max_docs=0):
        self.spec = normalize(spec)
        self.shard = (int(shard[0]), int(shard[1]))
        self.max_docs = int(max_docs)
        self.docs = 0
        self.lines = 0
        self._state = None
        self._ds = None

    def restore(self, state):
        self._state = state or None

    def state_dict(self):
        if self.spec["kind"] == "texts":
            return {"lines": self.lines, "docs": self.docs}
        try:
            hf = self._ds.state_dict() if self._ds is not None else None
        except Exception:  # nothing iterated yet, or the dataset has no state
            hf = None
        return {"hf": hf, "docs": self.docs}

    def __iter__(self):
        if self.spec["kind"] == "texts":
            return self._file_docs()
        return self._hf_docs()

    def _row_text(self, row, where):
        if not isinstance(row, dict):
            return None
        for column, value in where.items():
            got = row.get(column)
            if got is None or str(got).strip().lower() != str(value).strip().lower():
                return None
        parts = []
        for field in self.spec["fields"]:
            value = row.get(field)
            if not isinstance(value, str):
                raise SystemExit(f"field {field!r} of {describe(self.spec)} is not a string column")
            parts.append(value)
        return "\n".join(parts)

    def _hf_docs(self):
        import datasets
        from datasets.distributed import split_dataset_by_node

        source = self.spec
        if source["kind"] == "fineweb":
            ds = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        else:
            ds = datasets.load_dataset(source["id"], name=source["config"], split=source["split"], streaming=True)
        ds = split_dataset_by_node(ds, rank=self.shard[0], world_size=self.shard[1])
        if self._state:
            self.docs = int(self._state.get("docs") or 0)
            if self._state.get("hf"):
                ds.load_state_dict(self._state["hf"])
        self._ds = ds
        where = source.get("where") or {}
        for row in ds:
            if self.max_docs and self.docs >= self.max_docs:
                return
            text = self._row_text(row, where)
            if text is None:
                continue
            self.docs += 1
            yield text

    def _file_docs(self):
        path = Path(self.spec["path"])
        start = int((self._state or {}).get("lines") or 0)
        self.docs = int((self._state or {}).get("docs") or 0)
        where = self.spec.get("where") or {}
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for index, line in enumerate(f):
                if index < start:
                    continue
                if self.max_docs and self.docs >= self.max_docs:
                    return
                if index % self.shard[1] != self.shard[0]:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                if isinstance(row, dict):
                    text = self._row_text(row, where)
                elif isinstance(row, str):
                    text = row
                else:
                    text = line
                if text is None:
                    continue
                self.lines = index + 1
                self.docs += 1
                yield text
