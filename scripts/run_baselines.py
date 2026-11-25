"""CPU baselines -> predictions/*.jsonl

    python scripts/run_baselines.py            # majority, length_only, lexical_overlap (seconds)
    python scripts/run_baselines.py --nli      # DeBERTa NLI cross-encoder (minutes on a GPU; hours on a laptop, resumable)
    python scripts/run_baselines.py --nli --sample 0.3   # same, on a fixed random 30% of the rows
    add --external path/to/set.jsonl to also score an external test set
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjudge import baselines as B  # noqa: E402
from gjudge.data import stable_fraction  # noqa: E402
from gjudge.schema import load_jsonl, read_jsonl, write_jsonl  # noqa: E402

OUT = ROOT / "predictions"


def save(name, rows, scores, val, val_scores, latency, meta):
    preds = B.predictions(rows, scores, val, val_scores, latency)
    write_jsonl(OUT / f"{name}.jsonl", preds)
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(name, "done", flush=True)


def timed(fn, *args):
    t = time.time()
    out = fn(*args)
    return out, (time.time() - t) / len(args[-1])


def nli_scores(rows) -> tuple[list[float], float, str]:
    """Resumable: finished scores are kept in data/nli_cache.jsonl (id, score, seconds)."""
    cache_path = ROOT / "data" / "nli_cache.jsonl"
    cache = {c["id"]: c for c in read_jsonl(cache_path)} if cache_path.exists() else {}
    todo = [r for r in rows if r["id"] not in cache]
    # random order, so an interrupted run is still an unbiased sample
    todo.sort(key=lambda r: stable_fraction(r["id"], 17))
    model = B.NLI() if todo else None
    with open(cache_path, "a", encoding="utf-8", newline="\n") as f:
        for k, r in enumerate(todo):
            t = time.time()
            cache[r["id"]] = {"id": r["id"], "score": model.score(r), "s": time.time() - t,
                               "device": model.device_name}
            f.write(json.dumps(cache[r["id"]]) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"nli {k}/{len(todo)}", flush=True)
    mine = [cache[r["id"]] for r in rows]
    return [c["score"] for c in mine], sum(c["s"] for c in mine) / len(mine), mine[-1]["device"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--sample", type=float, default=1.0)
    ap.add_argument("--external", default=None, help="extra JSONL test set; default: every data/external/*.jsonl")
    a = ap.parse_args()
    D = ROOT / "data"
    train, val = load_jsonl(D / "train.jsonl"), load_jsonl(D / "val.jsonl")
    ext = [a.external] if a.external else sorted(glob.glob(str(D / "external" / "*.jsonl")))
    rows = val + load_jsonl(D / "test.jsonl") + load_jsonl(D / "transfer.jsonl")
    rows += [r for p in ext for r in load_jsonl(p)]
    n = len(val)
    if a.nli:
        rows = [r for r in rows if stable_fraction(r["id"], 17) < a.sample]
        val = [r for r in val if stable_fraction(r["id"], 17) < a.sample]
        s, lat, device = nli_scores(rows)
        save("nli_deberta_small", rows, s, val, s[:len(val)], lat,
             {"title": "NLI DeBERTa-v3-small (off the shelf)", "size_mb": 568, "device": device, "prob": True,
              "sample": a.sample})
        return
    save("majority", rows, [0.5] * len(rows), val, [0.5] * n, 0.0,
         {"title": "Majority class (always unsupported)", "size_mb": 0, "device": "-", "prob": False})
    s = B.length_scores(train, rows)
    save("length_only", rows, s, val, s[:n], 0.0,
         {"title": "Length only (shortcut probe)", "size_mb": 0, "device": "laptop CPU", "prob": True})
    s, lat = timed(B.lexical_scores, rows)
    save("lexical_overlap", rows, s, val, s[:n], lat,
         {"title": "Lexical overlap", "size_mb": 0, "device": "laptop CPU", "prob": False})


if __name__ == "__main__":
    main()
