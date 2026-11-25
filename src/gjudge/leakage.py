"""Duplicate and leakage checks between splits."""
import re
from collections import Counter, defaultdict

from .schema import context_text

K = 5  # words per shingle
MIN_CONTAINMENT = 0.5


def normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def row_key(row: dict) -> str:
    return normalise(row["answer"] + " | " + context_text(row))


def shingles(text: str) -> set[int]:
    w = normalise(text).split()
    return {hash(tuple(w[i:i + K])) for i in range(max(1, len(w) - K + 1))}


def near_duplicates(left: list[str], right: list[str], min_containment: float = MIN_CONTAINMENT):
    """Pairs (i, j, containment): share of the SMALLER text's 5-word shingles found in the other one.
    Containment, not Jaccard, so a short passage copied inside a long one is also caught."""
    rs = [shingles(t) for t in right]
    index = defaultdict(list)
    for j, s in enumerate(rs):
        for h in s:
            index[h].append(j)
    out = []
    for i, t in enumerate(left):
        s = shingles(t)
        shared = Counter(j for h in s for j in index.get(h, ()))
        for j, n in shared.items():
            c = n / min(len(s), len(rs[j]))
            if c >= min_containment:
                out.append((i, j, c))
    return out


def leaky_indices(rows: list[dict], protected: list[dict]) -> dict[int, str]:
    """Rows to drop, with the reason, because they overlap a split that must stay clean.
    Reasons: same source group, same answer+context, near-duplicate passage."""
    keys = {row_key(r) for r in protected}
    groups = {r["group"] for r in protected if "group" in r}
    bad: dict[int, str] = {}
    by_passage = defaultdict(list)  # compare unique passages: one article appears in up to six rows
    for i, r in enumerate(rows):
        if r.get("group") in groups:
            bad[i] = "same_group"
        elif row_key(r) in keys:
            bad[i] = "exact_row"
        for p in r["context"]:
            by_passage[p["text"]].append(i)
    texts = list(by_passage)
    right = sorted({p["text"] for r in protected for p in r["context"]})
    for i, _, _ in near_duplicates(texts, right):
        for k in by_passage[texts[i]]:
            bad.setdefault(k, "near_duplicate_passage")
    return bad


def exact_duplicates(rows: list[dict]) -> set[int]:
    """Indices of repeated answer+context rows after the first copy."""
    seen, dup = set(), set()
    for i, r in enumerate(rows):
        k = row_key(r)
        if k in seen:
            dup.add(i)
        seen.add(k)
    return dup


def overlap_report(a: list[dict], b: list[dict]) -> dict[str, int]:
    """Counts that must all be zero between two splits."""
    passages = lambda rs: {p["text"] for r in rs for p in r["context"]}
    return {
        "shared_groups": len({r["group"] for r in a if "group" in r} & {r["group"] for r in b if "group" in r}),
        "shared_rows": len({row_key(r) for r in a} & {row_key(r) for r in b}),
        "shared_passages": len({normalise(t) for t in passages(a)} & {normalise(t) for t in passages(b)}),
        "near_duplicate_passages": len(near_duplicates(sorted(passages(a)), sorted(passages(b)))),
    }
