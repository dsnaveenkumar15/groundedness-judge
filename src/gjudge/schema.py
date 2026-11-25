"""The data contract. One JSON object per line:
{"id", "question", "context": [{"id", "text"}], "answer", "label", "source", "split"}
Extra keys are allowed ("group" = source document, "evidence" = wrong span); outside data may lack them.
"""
import json
from pathlib import Path

LABELS = ("supported", "unsupported")


def validate(row: dict) -> dict:
    for key in ("id", "question", "answer", "label", "source", "split"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"row {row.get('id')!r}: {key!r} must be a string")
    if row["label"] not in LABELS:
        raise ValueError(f"row {row['id']!r}: bad label {row['label']!r}")
    if not row["answer"].strip():
        raise ValueError(f"row {row['id']!r}: empty answer")
    ctx = row.get("context")
    if not isinstance(ctx, list) or not ctx:
        raise ValueError(f"row {row['id']!r}: context must be a non-empty list")
    for p in ctx:
        if not isinstance(p.get("id"), str) or not isinstance(p.get("text"), str):
            raise ValueError(f"row {row['id']!r}: each passage needs string id and text")
    return row


def load_jsonl(path) -> list[dict]:
    """Load and validate one split or an external test set. Ids must be unique."""
    with open(path, encoding="utf-8") as f:
        rows = [validate(json.loads(line)) for line in f if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return rows


def read_jsonl(path) -> list[dict]:
    """Any JSONL file, no contract check (used for prediction files)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def context_text(row: dict) -> str:
    return "\n".join(p["text"] for p in row["context"])


def group_of(row: dict) -> str:
    """Rows from one source document share a group; the bootstrap resamples groups, not rows."""
    return row.get("group", row["id"])


def is_unsupported(rows) -> list[int]:
    return [int(r["label"] == "unsupported") for r in rows]
