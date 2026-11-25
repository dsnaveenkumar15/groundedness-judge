"""Load the small LLM, tokenise examples, and predict verdicts."""
import json
import random
import time
from pathlib import Path

import torch

from . import prompt
from .schema import write_jsonl

IGNORE = -100


def require_gpu() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU. In Colab: Runtime > Change runtime type > T4 GPU, then Run all again.")
    return torch.cuda.get_device_name(0)


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["revision"])


def bnb_config():
    from transformers import BitsAndBytesConfig
    # T4 has no bfloat16 hardware, so compute in float16.
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)


def load_model(cfg, quantize: bool = True, adapter=None):
    """Base model in 4-bit NF4 (or plain, for CPU checks), optionally with a trained LoRA adapter on top."""
    from transformers import AutoModelForCausalLM
    kw = {"revision": cfg["revision"], "device_map": {"": 0 if torch.cuda.is_available() else "cpu"}}
    if quantize:
        kw["quantization_config"] = bnb_config()
        kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def prompt_text(tok, row, shots=()) -> str:
    return tok.apply_chat_template(prompt.messages(row, shots), tokenize=False, add_generation_prompt=True)


def encode_prompt(tok, row, shots=()) -> list[int]:
    return tok(prompt_text(tok, row, shots), add_special_tokens=False)["input_ids"]


def encode_example(tok, row, max_tokens: int) -> dict:
    """Training example. Loss is on the verdict tokens and the end-of-turn token only; the prompt is masked."""
    p = encode_prompt(tok, row)
    c = tok(row["label"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p) + len(c) > max_tokens:
        raise ValueError(f"{row['id']}: {len(p) + len(c)} tokens > {max_tokens}; fix data prep, never truncate")
    return {"input_ids": p + c, "labels": [IGNORE] * len(p) + c}


def collate(features, pad_id: int) -> dict:
    """Left padding, so every verdict sits at the right edge and only those logits have to be computed."""
    n = max(len(f["input_ids"]) for f in features)
    pad = lambda key, value: torch.tensor([[value] * (n - len(f[key])) + list(f[key]) for f in features])
    mask = torch.tensor([[0] * (n - len(f["input_ids"])) + [1] * len(f["input_ids"]) for f in features])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": mask, "labels": pad("labels", IGNORE)}


def verdict_ids(tok) -> tuple[int, int]:
    """First token of each verdict word. They must differ, or the score would mean nothing."""
    s, u = (tok(w, add_special_tokens=False)["input_ids"][0] for w in ("supported", "unsupported"))
    if s == u:
        raise ValueError("verdict words share their first token")
    return s, u


def pick_shots(train_rows, n_per_label: int, seed: int, max_words: int = 150) -> list[dict]:
    """Short training rows from RAG-style sources, alternating labels. Never from validation or test."""
    rng = random.Random(seed)
    size = lambda r: len((r["answer"] + " ".join(p["text"] for p in r["context"])).split())
    out = []
    by_label = {lab: sorted((r for r in train_rows if r["label"] == lab and r["source"] != "vitaminc"
                             and size(r) <= max_words), key=lambda r: r["id"])
                for lab in ("unsupported", "supported")}
    picks = {lab: rng.sample(rows, n_per_label) for lab, rows in by_label.items()}
    for k in range(n_per_label):
        out += [picks["unsupported"][k], picks["supported"][k]]
    return out


def make_batches(lengths, batch_tokens: int) -> list[list[int]]:
    """Group similar lengths; (longest in batch) * (batch size) stays within batch_tokens."""
    order, out, cur = sorted(range(len(lengths)), key=lambda i: -lengths[i]), [], []
    for i in order:
        if cur and lengths[cur[0]] * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


@torch.inference_mode()
def predict(model, tok, rows, cfg, shots=(), log=print) -> list[dict]:
    """Greedy verdict + score. score = P(unsupported) against P(supported) at the first reply token.
    If the reply has no verdict word, the score decides and parsed=false is recorded."""
    pc = cfg["predict"]
    enc = [encode_prompt(tok, r, shots) for r in rows]
    for r, e in zip(rows, enc):
        if len(e) > pc["max_tokens"]:
            raise ValueError(f"{r['id']}: prompt has {len(e)} tokens > {pc['max_tokens']}")
    s_id, u_id = verdict_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out: list = [None] * len(rows)
    t0, done = time.time(), 0
    for batch in make_batches([len(e) for e in enc], pc["batch_tokens"]):
        n = max(len(enc[i]) for i in batch)
        ids = torch.tensor([[pad] * (n - len(enc[i])) + enc[i] for i in batch], device=model.device)
        mask = torch.tensor([[0] * (n - len(enc[i])) + [1] * len(enc[i]) for i in batch], device=model.device)
        t = time.time()
        g = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, temperature=None, top_p=None,
                           top_k=None, repetition_penalty=1.0, max_new_tokens=pc["max_new_tokens"],
                           pad_token_id=pad, output_logits=True, return_dict_in_generate=True)
        latency = (time.time() - t) / len(batch)
        first = g.logits[0].float()
        p = torch.stack([first[:, s_id], first[:, u_id]], -1).softmax(-1)[:, 1].tolist()
        for k, i in enumerate(batch):
            text = tok.decode(g.sequences[k, n:], skip_special_tokens=True).strip()
            verdict = prompt.parse_verdict(text)
            if p[k] != p[k]:
                raise RuntimeError(f"{rows[i]['id']}: score is NaN (float16 overflow?)")
            out[i] = {"id": rows[i]["id"], "pred": verdict or ("unsupported" if p[k] >= 0.5 else "supported"),
                      "score": p[k], "latency_s": latency, "parsed": verdict is not None, "raw": text}
        done += len(batch)
        if log and (done == len(rows) or done % 200 < len(batch)):
            log(f"  {done}/{len(rows)} in {time.time() - t0:.0f}s")
    return out


def save_predictions(out_dir, name: str, preds: list[dict], meta: dict) -> None:
    """predictions/<name>.jsonl + <name>.meta.json. Only ids, verdicts and scores: no dataset text."""
    out_dir = Path(out_dir)
    write_jsonl(out_dir / f"{name}.jsonl", preds)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
