"""Read local data without changing record contents."""

import gzip
import json
from contextlib import contextmanager
from pathlib import Path


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            return [json.loads(line) for line in stream if line.strip()]
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array or JSONL records.")
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def jsonl_writer(path):
    """Save each completed record so an interrupted API run retains its work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        def emit(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
        yield emit


def corpus_text(row):
    if isinstance(row, str):
        return row
    for key in ("text", "patient", "content"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    raise ValueError("Corpus record needs text, patient, or content.")
