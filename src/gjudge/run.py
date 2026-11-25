"""Command line for the GPU steps. The notebook only calls these commands, each in its own process,
so GPU memory is freed between steps and pinned package versions need no kernel restart.

    python -m gjudge.run predict --system qwen_zero_shot
    python -m gjudge.run predict --system qwen_few_shot --shots 4
    python -m gjudge.run train --r 16 --out runs/sft_r16
    python -m gjudge.run predict --system qwen_sft_r16 --adapter runs/sft_r16/adapter
"""
import argparse
import glob
import json
import os
from pathlib import Path

import torch

from . import llm, train
from .evaluate import EXPECTED
from .schema import load_jsonl, read_jsonl


def load_train(data_dir: Path, external_paths) -> list[dict]:
    """The only place training rows are read. External rows may never be in them."""
    rows = load_jsonl(data_dir / "train.jsonl")
    if any(r["split"] != "train" for r in rows):
        raise ValueError("train.jsonl holds rows of another split")
    ext = {r["id"] for p in external_paths for r in load_jsonl(p)}
    if ext & {r["id"] for r in rows}:
        raise ValueError("external test rows found in train.jsonl")
    return rows


def base_predictions_exist(system: str, cfg, rows) -> bool:
    """Untrained-model predictions do not change, so a later session (r=8, DPO) need not repeat them."""
    preds, meta = Path(f"predictions/{system}.jsonl"), Path(f"predictions/{system}.meta.json")
    return (preds.exists() and meta.exists()
            and json.loads(meta.read_text(encoding="utf-8"))["base_model"] == cfg["base_model"]
            and {r["id"] for r in rows} <= {p["id"] for p in read_jsonl(preds)})


def size_mb(path) -> float:
    return round(sum(p.stat().st_size for p in Path(path).rglob("*.safetensors")) / 1e6, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("train", "dpo", "predict", "merge"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", default="configs/qlora.json")
        p.add_argument("--data", default="data")
        p.add_argument("--out", default=None)
        p.add_argument("--adapter", default=None)
        p.add_argument("--r", type=int, default=None)
        p.add_argument("--system", default=None)
        p.add_argument("--shots", type=int, default=0)
    a = ap.parse_args(argv)
    if not os.environ.get("GJ_ALLOW_CPU"):  # set only by the local smoke test
        print("GPU:", llm.require_gpu())
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    D = Path(a.data)
    ext = sorted(glob.glob(str(D / "external" / "*.jsonl")))

    if a.cmd == "merge":
        train.merge_adapter(cfg, a.adapter, a.out)
        return print("merged model in", a.out)
    if a.cmd == "predict":
        rows = [r for n in ("val", "test", "transfer") for r in load_jsonl(D / f"{n}.jsonl")]
        rows += [r for p in ext for r in load_jsonl(p)]
        if not a.adapter and base_predictions_exist(a.system, cfg, rows):
            return print(a.system, "already complete, skipped")
    tok = llm.load_tokenizer(cfg)
    model = llm.load_model(cfg, quantize=True, adapter=a.adapter)
    if a.cmd == "train":
        info = train.train_sft(cfg, model, tok, load_train(D, ext), load_jsonl(D / "val.jsonl"), a.out, r=a.r)
    elif a.cmd == "dpo":
        info = train.train_dpo(cfg, model, tok, load_train(D, ext), a.out)
    else:
        shots = llm.pick_shots(load_train(D, ext), a.shots // 2, cfg["seed"]) if a.shots else ()
        preds = llm.predict(model, tok, rows, cfg, shots)
        base_mb = round(model.get_memory_footprint() / 1e6)
        meta = {"title": dict(EXPECTED).get(a.system, a.system),
                "base_model": cfg["base_model"], "prob": True, "size_mb": base_mb, "shots": [s["id"] for s in shots],
                "adapter_mb": size_mb(a.adapter) if a.adapter else 0,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None}
        llm.save_predictions("predictions", a.system, preds, meta)
        return print("wrote predictions for", a.system, meta)
    print({k: v for k, v in info.items() if k != "history"})


if __name__ == "__main__":
    main()
