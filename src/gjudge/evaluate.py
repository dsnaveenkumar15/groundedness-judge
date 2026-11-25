"""Read predictions of every system and write docs/results.md.

    python -m gjudge.evaluate [--external radar.jsonl] [--root .]

Rules, same for all systems:
- the cut-off on the score is chosen on VALIDATION only (best balanced accuracy), then frozen;
- a system without full validation scores is judged on its own verdicts;
- confidence intervals resample source documents, not rows.
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import leakage, metrics as M
from .schema import context_text, group_of, is_unsupported, load_jsonl, read_jsonl

# Shown as "not run" until a predictions file exists. Any other predictions/*.jsonl is added at the end.
EXPECTED = [
    ("majority", "Majority class (always unsupported)"),
    ("length_only", "Length only (shortcut probe)"),
    ("lexical_overlap", "Lexical overlap"),
    ("nli_deberta_small", "NLI DeBERTa-v3-small (off the shelf)"),
    ("qwen_zero_shot", "Qwen2.5-1.5B base, zero-shot"),
    ("qwen_few_shot", "Qwen2.5-1.5B base, 4-shot"),
    ("qwen_sft_r16", "Qwen2.5-1.5B + QLoRA SFT r=16"),
    ("qwen_sft_r8", "Qwen2.5-1.5B + QLoRA SFT r=8 (ablation)"),
    ("qwen_dpo_r16", "Qwen2.5-1.5B + QLoRA DPO r=16 (optional)"),
    ("paid_judge", "Paid judge model"),
]
PAIRS = [("qwen_sft_r16", "qwen_zero_shot"), ("qwen_sft_r16", "qwen_few_shot"), ("qwen_sft_r16", "nli_deberta_small"),
         ("qwen_sft_r16", "lexical_overlap"), ("qwen_sft_r16", "qwen_sft_r8"), ("qwen_sft_r16", "qwen_dpo_r16"),
         ("qwen_sft_r16", "paid_judge"), ("nli_deberta_small", "lexical_overlap")]


def bacc(y, p):
    return M.classification(y, p)["balanced_accuracy"]


def f1(y, p):
    return M.classification(y, p)["f1"]


def load_system(pred_dir: Path, name: str) -> dict | None:
    path = pred_dir / f"{name}.jsonl"
    if not path.exists():
        return None
    meta_path = pred_dir / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    preds = {}
    for p in read_jsonl(path):
        if p["pred"] not in ("supported", "unsupported"):
            raise ValueError(f"{path}: bad pred {p['pred']!r}")
        if p["id"] in preds:
            raise ValueError(f"{path}: duplicate id {p['id']}")
        preds[p["id"]] = p
    return {"name": name, "title": meta.get("title", name), "meta": meta, "preds": preds}


def threshold_for(system, val, min_rows: int = 100) -> float | None:
    """Validation-chosen cut-off. Needs scores for all validation rows, or for at least `min_rows` of them
    (slow systems may be run on a random sample). Otherwise None."""
    rows = [r for r in val if system["preds"].get(r["id"], {}).get("score") is not None]
    if not rows or (len(rows) < len(val) and len(rows) < min_rows):
        return None
    return M.best_threshold(is_unsupported(rows), [system["preds"][r["id"]]["score"] for r in rows])


def decide(system, rows, threshold):
    """(pred array, score array or None). Missing rows are an error: no system may skip hard items."""
    missing = [r["id"] for r in rows if r["id"] not in system["preds"]]
    if missing:
        raise ValueError(f"{system['name']}: no prediction for {len(missing)} rows, e.g. {missing[0]}")
    ps = [system["preds"][r["id"]] for r in rows]
    scores = [p.get("score") for p in ps]
    has = all(s is not None for s in scores)
    if threshold is not None and has:
        return np.array([int(s >= threshold) for s in scores]), np.array(scores)
    return np.array([int(p["pred"] == "unsupported") for p in ps]), np.array(scores) if has else None


def covers(system, rows) -> bool:
    return all(r["id"] in system["preds"] for r in rows)


def ci(x, lo_hi) -> str:
    return f"{x:.3f} ({lo_hi[0]:.3f}-{lo_hi[1]:.3f})"


def result_row(system, rows, threshold, n_boot, store: dict) -> str:
    y, g = is_unsupported(rows), [group_of(r) for r in rows]
    p, s = decide(system, rows, threshold)
    c = M.classification(y, p)
    store[system["name"]] = {**c, "threshold": threshold, "auroc": M.auroc(y, s) if s is not None else None,
                             "ece": M.ece(y, s) if s is not None and system["meta"].get("prob") else None,
                             "balanced_accuracy_ci95": M.bootstrap_ci(bacc, g, y, p, n=n_boot)}
    auc = ci(M.auroc(y, s), M.bootstrap_ci(M.auroc, g, y, s, n=n_boot)) if s is not None else "-"
    cal = f"{M.ece(y, s):.3f}" if s is not None and system["meta"].get("prob") else "-"
    lat = np.mean([system["preds"][r["id"]].get("latency_s") or 0.0 for r in rows]) * 1000
    size = system["meta"].get("size_mb")
    return (f"| {system['title']} | {ci(c['balanced_accuracy'], M.bootstrap_ci(bacc, g, y, p, n=n_boot))} "
            f"| {c['precision']:.3f} | {c['recall']:.3f} | {ci(c['f1'], M.bootstrap_ci(f1, g, y, p, n=n_boot))} "
            f"| {auc} | {cal} | {lat:.1f} ({system['meta'].get('device', '?')}) | {'-' if size is None else size} |")


HEADER = ("| System | Balanced accuracy (95% CI) | Precision | Recall | F1 (95% CI) | AUROC (95% CI) | ECE "
          "| ms per item (device) | Size MB |\n|---|---|---|---|---|---|---|---|---|")


def table(systems, rows, thresholds, n_boot, store: dict) -> list[str]:
    out = [HEADER]
    for name, title, system in systems:
        if system is None or not covers(system, rows):
            state = "not run" if system is None or not sampled(system, rows) else "sample only, see below"
            out.append(f"| {title} | {state} | | | | | | | |")
            store[name] = state
        else:
            out.append(result_row(system, rows, thresholds[name], n_boot, store))
    return out


def sampled(system, rows) -> list[dict]:
    """Rows this system answered, when that is a real part of them but not all."""
    part = [r for r in rows if r["id"] in system["preds"]]
    return part if len(part) < len(rows) and len({r["label"] for r in part}) == 2 else []


def sample_section(systems, sets: dict, thresholds, n_boot, numbers) -> list[str]:
    """Same tables again on the random subset that a slow system was run on, for every system that covers it."""
    out = []
    for name, title, system in systems:
        for set_name, rows in sets.items():
            part = sampled(system, rows) if system is not None else []
            if part:
                key = f"{set_name}, only the rows {name} was run on"
                out += ["", f"### {key}", "", describe(part), ""]
                have = [x for x in systems if x[2] is not None and covers(x[2], part)]
                out += table(have, part, thresholds, n_boot, numbers.setdefault(key, {}))
    return out


def describe(rows) -> str:
    n = Counter(r["label"] for r in rows)
    return (f"{len(rows)} rows from {len({group_of(r) for r in rows})} source documents: "
            f"{n['supported']} supported, {n['unsupported']} unsupported.")


def by_source(systems, rows, thresholds, n_boot) -> list[str]:
    sources = sorted({r["source"] for r in rows})
    out = ["| System | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for name, title, system in systems:
        if system is None or not covers(system, rows):
            continue
        cells = []
        for s in sources:
            part = [r for r in rows if r["source"] == s]
            y, p = is_unsupported(part), decide(system, part, thresholds[name])[0]
            cells.append(ci(bacc(y, p), M.bootstrap_ci(bacc, [group_of(r) for r in part], y, p, n=n_boot)))
        out.append(f"| {title} | " + " | ".join(cells) + " |")
    return out


def own_verdicts(systems, rows) -> list[str]:
    """LLM verdicts as generated, without the validation cut-off, plus how often the reply had no verdict word."""
    out = ["| System | Balanced accuracy | Says unsupported | Reply not parseable |", "|---|---|---|---|"]
    for name, title, system in systems:
        if system is None or not covers(system, rows):
            continue
        ps = [system["preds"][r["id"]] for r in rows]
        if "parsed" not in ps[0]:
            continue
        p = [int(x["pred"] == "unsupported") for x in ps]
        bad = sum(not x["parsed"] for x in ps)
        out.append(f"| {title} | {bacc(is_unsupported(rows), p):.3f} | {np.mean(p):.1%} | {bad} of {len(ps)} |")
    return out if len(out) > 2 else ["No LLM system has been run yet."]


def mcnemar_rows(systems, rows, thresholds) -> list[str]:
    have = {n: s for n, _, s in systems if s is not None and covers(s, rows)}
    y, out = is_unsupported(rows), ["| A | B | only A right | only B right | p (exact) |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        if a in have and b in have:
            m = M.mcnemar(y, decide(have[a], rows, thresholds[a])[0], decide(have[b], rows, thresholds[b])[0])
            out.append(f"| {a} | {b} | {m['only_a_right']} | {m['only_b_right']} | {m['p_value']:.4f} |")
        else:
            out.append(f"| {a} | {b} | not run | | |")
    return out


def reliability_tables(systems, rows) -> list[str]:
    out = []
    for name, title, system in systems:
        if system is None or not covers(system, rows) or not system["meta"].get("prob"):
            continue
        s = [system["preds"][r["id"]]["score"] for r in rows]
        out += [f"\n**{title}**\n", "| Score bin | n | Mean score | Really unsupported |", "|---|---|---|---|"]
        out += [f"| {b:.1f}-{b + 0.1:.1f} | {n} | {conf:.2f} | {rate:.2f} |"
                for b, n, conf, rate in M.reliability(is_unsupported(rows), s)]
    return out or ["No system with probability scores has been run yet."]


def error_analysis(system, rows, threshold, full_path: Path, per_kind: int = 10) -> list[str]:
    """Counts and ids for the committed file; full text goes to a gitignored file (source texts are not ours)."""
    y, (p, s) = is_unsupported(rows), decide(system, rows, threshold)
    kinds = {"false supported (missed hallucination)": [i for i in range(len(rows)) if y[i] and not p[i]],
             "false unsupported (false alarm)": [i for i in range(len(rows)) if not y[i] and p[i]]}
    out, full = [], [f"# Errors of {system['title']} (do not publish: contains dataset text)"]
    for kind, idx in kinds.items():
        per_source = Counter(rows[i]["source"] for i in idx)
        hard = sum(bool(rows[i].get("hard")) for i in idx)
        out.append(f"- {kind}: {len(idx)} ({', '.join(f'{k} {v}' for k, v in sorted(per_source.items()))}; "
                   f"{hard} marked 'true but not in the source' by the annotators)")
        # most confident mistakes first
        conf = (lambda i: -abs(s[i] - (threshold or 0.5))) if s is not None else (lambda i: i)
        pick = sorted(idx, key=conf)[:per_kind]
        out.append(f"  - most confident: {', '.join(rows[i]['id'] for i in pick)}")
        for i in pick:
            r = rows[i]
            full.append(f"\n## {kind}: {r['id']}\n\nscore {s[i] if s is not None else '-'}\n\n**Q** {r['question']}\n\n"
                        f"**Context** {context_text(r)}\n\n**Answer** {r['answer']}\n\n**Marked span** {r.get('evidence', '')}")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text("\n".join(full), encoding="utf-8")
    return out


def build(root: Path, externals=(), n_boot: int = 1000) -> tuple[str, dict]:
    D = root / "data"
    val, test, transfer = (load_jsonl(D / f"{n}.jsonl") for n in ("val", "test", "transfer"))
    pred_dir = root / "predictions"
    names = [n for n, _ in EXPECTED]
    extra = sorted(p.stem for p in pred_dir.glob("*.jsonl") if p.stem not in names)
    systems = []
    for name, title in EXPECTED + [(n, n) for n in extra]:
        system = load_system(pred_dir, name)
        systems.append((name, system["title"] if system else title, system))
    thresholds = {n: threshold_for(s, val) for n, _, s in systems if s is not None}
    numbers: dict = {}

    L = ["# Results", "", "Written by `python -m gjudge.evaluate`. Do not edit by hand.", "",
         "Positive class = unsupported. Cut-offs were chosen on validation only. 95% CIs: bootstrap over source "
         f"documents, {n_boot} resamples. Both test sets are balanced, so precision and F1 do not carry over to live "
         "traffic where unsupported answers are rarer; balanced accuracy, recall and AUROC do.", "",
         "Latency is batched throughput on the named device, so compare it only within one device.", "",
         "## Main test", "", describe(test), ""]
    L += table(systems, test, thresholds, n_boot, numbers.setdefault("test", {}))
    L += ["", "### Balanced accuracy by source", ""] + by_source(systems, test, thresholds, n_boot)
    L += ["", "## Transfer test: RAG question answering (never trained on)", "",
          describe(transfer) + " MS MARCO passages from RAGTruth; held out on purpose, see docs/data.md.", ""]
    L += table(systems, transfer, thresholds, n_boot, numbers.setdefault("transfer", {}))
    part = sample_section(systems, {"Main test": test, "Transfer test": transfer}, thresholds, n_boot, numbers)
    if part:
        L += ["", "## Sampled comparisons", "", "Some systems answered only part of a test set. The NLI part is a random one, fixed by a hash of the row id. "
              "Every system that covers the same rows is shown for comparison."] + part
    L += ["", "## External test: human-labelled answers from a real RAG app", ""]
    if not externals:
        L.append("not run (no external file given)")
    for path in externals:
        ext = load_jsonl(path)
        train = load_jsonl(D / "train.jsonl") if (D / "train.jsonl").exists() else []
        L += [f"File `{Path(path).name}`. " + describe(ext) + " Small set: read the intervals, not the point values.", "",
              f"Overlap with train (must be 0): {leakage.overlap_report(ext, train)}", ""]
        L += table(systems, ext, thresholds, n_boot, numbers.setdefault(f"external:{Path(path).name}", {}))
    L += ["", "## LLM verdicts as generated (main test, no tuned cut-off)", ""] + own_verdicts(systems, test)
    L += ["", "## Paired comparison (main test, McNemar exact test)", "",
          "p values are not corrected for several comparisons, and rows that share a document are not fully "
          "independent, so treat p between 0.01 and 0.05 as weak evidence.", ""]
    L += mcnemar_rows(systems, test, thresholds)
    L += ["", "## Calibration (main test)", "",
          "ECE in the table above is the gap between the score and the real rate."] + reliability_tables(systems, test)
    L += ["", "## Error analysis (main test)", "",
          "Full texts: data/error_analysis_<system>.md (local only)."]
    for name, title, system in systems:
        if system is not None and covers(system, test) and name in (
                "nli_deberta_small", "qwen_zero_shot", "qwen_few_shot", "qwen_sft_r16", "paid_judge"):
            L += ["", f"### {title}", ""] + error_analysis(
                system, test, thresholds[name], D / f"error_analysis_{name}.md")
    return "\n".join(L) + "\n", numbers


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--external", action="append", help="external JSONL; default: every data/external/*.jsonl")
    a = ap.parse_args(argv)
    root = Path(a.root)
    ext = a.external or sorted(glob.glob(str(root / "data" / "external" / "*.jsonl")))
    text, numbers = build(root, ext)
    (root / "docs").mkdir(exist_ok=True)
    for name, body in (("results.md", text), ("results.json", json.dumps(numbers, indent=1))):
        (root / "docs" / name).write_text(body, encoding="utf-8", newline="\n")
    print("wrote", root / "docs" / "results.md")


if __name__ == "__main__":
    main()
