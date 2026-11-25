"""Download public datasets (pinned), convert to the contract, split by source document, remove leaks, report.

    python scripts/prepare_data.py

Writes data/{train,val,test,transfer}.jsonl (gitignored) and docs/data_manifest.json (committed).
"""
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gjudge import data, leakage, prompt  # noqa: E402
from gjudge.schema import load_jsonl, read_jsonl, write_jsonl  # noqa: E402

SEED = 17
MAX_TOKENS = 1536   # prompt must fit whole; longer items are dropped, nothing is ever truncated
MARGIN = 40         # chat template tokens + verdict
RAW = ROOT / "data" / "raw"
HF = "https://huggingface.co/datasets"
RAGTRUTH = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset"
HAGRID = f"{HF}/miracl/hagrid/resolve/b2a085913606be3c4f2f1a8bff1810e38bade8fa/hagrid-v1.0-en"
VITAMINC = f"{HF}/tals/vitaminc/resolve/be6febb761b0b2807687e61e0b5282e459df2fa0"
FILES = {
    "ragtruth_response.jsonl": f"{RAGTRUTH}/response.jsonl",
    "ragtruth_source.jsonl": f"{RAGTRUTH}/source_info.jsonl",
    "hagrid_train.jsonl": f"{HAGRID}/train.jsonl",
    "hagrid_dev.jsonl": f"{HAGRID}/dev.jsonl",
    "vitaminc_train.jsonl": f"{VITAMINC}/train.jsonl",
    "vitaminc_dev.jsonl": f"{VITAMINC}/dev.jsonl",
    "vitaminc_test.jsonl": f"{VITAMINC}/test.jsonl",
}
TOKENIZER = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
# Rows per label and source. None = take all that are left after the filters.
CAPS = {
    "train": {"ragtruth_summary": None, "hagrid": None, "vitaminc": 500},
    "val": {"ragtruth_summary": 125, "hagrid": 60, "vitaminc": 60},
    "test": {"ragtruth_summary": None, "hagrid": None, "vitaminc": 80},
    "transfer": {"ragtruth_qa": None},
}


def download() -> dict[str, str]:
    RAW.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, url in FILES.items():
        path = RAW / name
        if not path.exists():
            print("download", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "dataset-download"})
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def token_counter():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1]))
    return lambda rows: [len(e.ids) + MARGIN for e in
                         tok.encode_batch([prompt.SYSTEM + prompt.user_message(r) for r in rows])]


def pools() -> dict[str, list[dict]]:
    """Assign every row to a split by its source document, before any filtering."""
    raw = lambda n: read_jsonl(RAW / n)
    resp, src = raw("ragtruth_response.jsonl"), raw("ragtruth_source.jsonl")
    out = {"train": [], "val": [], "test": [], "transfer": []}
    for r in data.ragtruth_rows(resp, src, "Summary"):  # official test stays test; val is cut from official train
        is_val = data.stable_fraction(r["group"], SEED) < 0.15
        out["test" if r["split"] == "test" else "val" if is_val else "train"].append(r)
    out["transfer"] = [r for r in data.ragtruth_rows(resp, src, "QA") if r["split"] == "test"]
    for r in data.hagrid_rows(raw("hagrid_train.jsonl") + raw("hagrid_dev.jsonl")):
        f = data.stable_fraction(r["group"], SEED)
        out["test" if f < 0.35 else "val" if f < 0.47 else "train"].append(r)
    for split, name in (("train", "train"), ("val", "dev"), ("test", "test")):  # official VitaminC splits
        out[split] += data.one_pair_per_page(
            data.vitaminc_rows(raw(f"vitaminc_{name}.jsonl"), split), SEED)
    # VitaminC has ~22k pages; thin the train pool early so the near-duplicate check stays small
    keep = lambda r: r["source"] != "vitaminc" or data.stable_fraction(r["group"], SEED) < 0.10
    out["train"] = [r for r in out["train"] if keep(r)]
    for split, rows in out.items():
        for r in rows:
            r["split"] = split
    return out


def count(rows) -> dict:
    return {f"{s}/{lab}": n for (s, lab), n in sorted(Counter((r["source"], r["label"]) for r in rows).items())}


def main():
    hashes = download()
    out, report = pools(), {}
    tokens = token_counter()
    for split, rows in out.items():
        lens = tokens(rows)
        fit = [r for r, n in zip(rows, lens) if n <= MAX_TOKENS]
        dup = leakage.exact_duplicates(fit)
        report[split] = {"pool": count(rows), "fits_max_tokens": count(fit), "exact_duplicates_dropped": len(dup)}
        out[split] = [r for i, r in enumerate(fit) if i not in dup]

    protected = out["test"] + out["transfer"]
    for split in ("val", "train"):
        bad = leakage.leaky_indices(out[split], protected)
        report[split]["leaks_dropped"] = dict(sorted(Counter(f"{why}/{out[split][i]['source']}" for i, why in bad.items()).items()))
        out[split] = [r for i, r in enumerate(out[split]) if i not in bad]
        protected = protected + out[split]

    for split, caps in CAPS.items():
        rows = [r for s, cap in caps.items()
                for r in data.balance([r for r in out[split] if r["source"] == s], SEED, cap)]
        out[split] = sorted(rows, key=lambda r: r["id"])
        lens = tokens(out[split])
        write_jsonl(ROOT / "data" / f"{split}.jsonl", out[split])
        load_jsonl(ROOT / "data" / f"{split}.jsonl")  # contract check
        report[split].update({
            "final": count(out[split]), "rows": len(out[split]), "groups": len({r["group"] for r in out[split]}),
            "tokens_mean": round(sum(lens) / len(lens)), "tokens_max": max(lens), "tokens_total": sum(lens),
            "sha256": hashlib.sha256((ROOT / "data" / f"{split}.jsonl").read_bytes()).hexdigest(),
        })

    pairs = [("train", "val"), ("train", "test"), ("train", "transfer"), ("val", "test"), ("val", "transfer")]
    overlaps = {f"{a}-{b}": leakage.overlap_report(out[a], out[b]) for a, b in pairs}
    manifest = {"seed": SEED, "max_tokens": MAX_TOKENS, "raw_sha256": hashes, "splits": report, "overlaps": overlaps}
    (ROOT / "docs" / "data_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"splits": report, "overlaps": overlaps}, indent=1))
    if any(v for o in overlaps.values() for v in o.values()):
        sys.exit("LEAKAGE: splits overlap, see above")


if __name__ == "__main__":
    main()
