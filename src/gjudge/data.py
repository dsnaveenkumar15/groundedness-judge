"""Turn raw public datasets into contract rows. Pure functions, no network."""
import hashlib
import random
import re
from collections import defaultdict


def stable_fraction(key: str, seed: int) -> float:
    """Deterministic number in [0, 1) for a key. Used to split by source document."""
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def ragtruth_rows(responses, sources, task: str) -> list[dict]:
    """task: "Summary" (used for train/val/test) or "QA" (MS MARCO text: evaluation only, never trained on).
    Any human-marked hallucination span, including "implicit_true", makes the answer unsupported."""
    by_id = {s["source_id"]: s for s in sources}
    out = []
    for r in responses:
        s = by_id[r["source_id"]]
        if s["task_type"] != task or r["quality"] != "good":
            continue
        if task == "QA":
            question, context = s["source_info"]["question"], split_passages(s["source_info"]["passages"])
        else:
            question, context = "Summarize the following news article.", [{"id": "1", "text": s["source_info"]}]
        spans = r["labels"]
        out.append({
            "id": f"ragtruth-{task.lower()}-{r['id']}", "question": question, "context": context,
            "answer": r["response"].strip(), "label": "unsupported" if spans else "supported",
            "source": f"ragtruth_{task.lower()}", "split": r["split"], "group": f"ragtruth-{r['source_id']}",
            "evidence": spans[0]["text"] if spans else "",
            "hard": bool(spans) and all(x.get("implicit_true") for x in spans),
        })
    return out


def split_passages(text: str) -> list[dict]:
    """'passage 1:...\\n\\npassage 2:...' -> contract context list."""
    parts = re.split(r"passage (\d+):", text)
    return [{"id": parts[i], "text": parts[i + 1].strip()} for i in range(1, len(parts) - 1, 2)]


def hagrid_rows(records) -> list[dict]:
    """Human attributability labels on GPT-3.5 answers. Unlabelled answers are dropped.
    Queries that cite the same Wikipedia article are joined into one group (see group_by_articles)."""
    groups = group_by_articles({d["query_id"]: [q["docid"].split("#")[0] for q in d["quotes"]] for d in records})
    out = []
    for d in records:
        context = [{"id": str(i), "text": q["text"]} for i, q in enumerate(d["quotes"], 1)]
        for k, a in enumerate(d["answers"]):
            if a.get("attributable") is None or not a["answer"].strip():
                continue
            bad = [s["text"] for s in a["sentences"] if s.get("attributable") == 0]
            out.append({
                "id": f"hagrid-{d['query_id']}-{k}", "question": d["query"], "context": context,
                "answer": a["answer"].strip(), "label": "supported" if a["attributable"] else "unsupported",
                "source": "hagrid", "split": "", "group": f"hagrid-{groups[d['query_id']]}",
                "evidence": bad[0] if bad else "",
            })
    return out


def group_by_articles(articles: dict) -> dict:
    """Union-find: queries that share an article id end up with the same group id."""
    parent = {q: q for q in articles}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    owner: dict = {}
    for q, arts in articles.items():
        for a in arts:
            if a in owner:
                parent[find(q)] = find(owner[a])
            else:
                owner[a] = q
    return {q: find(q) for q in articles}


def vitaminc_rows(records, split: str) -> list[dict]:
    """Real Wikipedia revisions only. SUPPORTS -> supported; REFUTES and NOT ENOUGH INFO -> unsupported."""
    out = []
    for d in records:
        if d["revision_type"] != "real":
            continue
        out.append({
            "id": f"vitaminc-{d['unique_id']}", "question": "Is this claim supported by the passage?",
            "context": [{"id": "1", "text": d["evidence"]}], "answer": d["claim"],
            "label": "supported" if d["label"] == "SUPPORTS" else "unsupported",
            "source": "vitaminc", "split": split, "group": f"vitaminc-{d['page']}", "evidence": d["claim"],
        })
    return out


def one_pair_per_page(rows, seed: int) -> list[dict]:
    """Per VitaminC page keep one supported and one unsupported row (when both exist), so no page dominates."""
    pages = defaultdict(lambda: {"supported": [], "unsupported": []})
    for r in rows:
        pages[r["group"]][r["label"]].append(r)
    rng, out = random.Random(seed), []
    for page in sorted(pages):
        for part in pages[page].values():
            if part:
                out.append(rng.choice(sorted(part, key=lambda r: r["id"])))
    return out


def balance(rows, seed: int, cap_per_label: int | None = None) -> list[dict]:
    """Same number of supported and unsupported rows, drawn at random with a fixed seed."""
    rng = random.Random(seed)
    parts = {lab: sorted((r for r in rows if r["label"] == lab), key=lambda r: r["id"])
             for lab in ("supported", "unsupported")}
    n = min(len(p) for p in parts.values())
    if cap_per_label is not None:
        n = min(n, cap_per_label)
    return [r for p in parts.values() for r in rng.sample(p, n)]
