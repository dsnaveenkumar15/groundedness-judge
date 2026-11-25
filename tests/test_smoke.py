"""The real training and prediction code on CPU with a tiny RANDOM Qwen2 and the real Qwen tokenizer.
Proves plumbing, not quality. Needs the tokenizer (about 11 MB, downloaded once)."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "qlora.json").read_text())
llm = pytest.importorskip("gjudge.llm")


@pytest.fixture(scope="module")
def tok():
    try:
        return llm.load_tokenizer(CFG)
    except OSError as e:
        pytest.skip(f"tokenizer not available offline: {e}")


@pytest.fixture(scope="module")
def tiny(tok, tmp_path_factory):
    """Save a random 2-layer Qwen2 to disk and return a config that points to it."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny")
    c = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                    tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    Qwen2ForCausalLM(c).save_pretrained(path)
    tok.save_pretrained(path)
    cfg = copy.deepcopy(CFG)
    cfg.update(base_model=str(path), revision=None)
    cfg["train"].update(epochs=1, batch_size=2, grad_accum=2, eval_steps=2, lr=0.01)
    cfg["dpo"].update(max_pairs=8, batch_size=2, grad_accum=2)
    return cfg


def rows(n, split="train"):
    return [{"id": f"{split}{i}", "question": "Who won?", "answer": f"Team {i} won the cup.", "split": split,
             "context": [{"id": "1", "text": f"Team {i if i % 2 else i + 50} won the cup. " * (1 + i % 3)}],
             "label": "supported" if i % 2 else "unsupported", "source": "s", "group": f"{split}{i}",
             "evidence": f"Team {i}"} for i in range(n)]


def test_verdicts_are_single_distinct_tokens(tok):
    s, u = llm.verdict_ids(tok)
    assert tok.decode([s]) == "supported" and tok.decode([u]) == "unsupported"


def test_label_mask_covers_only_the_verdict(tok):
    for r in rows(4):
        e = llm.encode_example(tok, r, 1600)
        kept = [t for t in e["labels"] if t != llm.IGNORE]
        assert len(e["labels"]) == len(e["input_ids"])
        assert tok.decode(kept) == r["label"] + "<|im_end|>"
        assert e["input_ids"][:-2] == llm.encode_prompt(tok, r)     # same prompt in training and prediction
        assert e["labels"][:-2] == [llm.IGNORE] * (len(e["labels"]) - 2)
    with pytest.raises(ValueError, match="never truncate"):
        llm.encode_example(tok, rows(1)[0], 20)


def test_collate_left_pads(tok):
    f = [llm.encode_example(tok, r, 1600) for r in rows(3)]
    b = llm.collate(f, tok.pad_token_id)
    assert (b["labels"][:, -2:] != llm.IGNORE).all() and (b["labels"][:, :-2] == llm.IGNORE).all()
    assert b["attention_mask"].sum(1).tolist() == [len(x["input_ids"]) for x in f]


def test_verdict_loss_equals_the_library_loss(tok, tiny):
    """Our logits_to_keep shortcut must give exactly the loss the model itself computes on full labels."""
    from gjudge.train import VerdictTrainer
    model = llm.load_model(tiny, quantize=False).float()
    b = llm.collate([llm.encode_example(tok, r, 1600) for r in rows(3)], tok.pad_token_id)
    with torch.no_grad():
        ours = VerdictTrainer.compute_loss(None, model, b)
        theirs = model(**b).loss
    assert ours.item() == pytest.approx(theirs.item(), rel=1e-5)


def test_batches_cover_everything_within_budget():
    lengths = [10, 900, 300, 300, 50, 5000]
    batches = llm.make_batches(lengths, 1000)
    assert sorted(i for b in batches for i in b) == list(range(6))
    assert all(len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 1000 for b in batches)
    assert llm.make_batches([], 10) == []


def test_few_shot_only_takes_short_balanced_rows():
    train = rows(20)
    shots = llm.pick_shots(train, 2, seed=1)
    assert [s["label"] for s in shots] == ["unsupported", "supported"] * 2
    assert shots == llm.pick_shots(train, 2, seed=1)


def test_require_gpu_fails_loudly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="T4"):
        llm.require_gpu()


def test_sft_train_resume_and_predict(tok, tiny, tmp_path):
    from gjudge.train import train_sft
    out = tmp_path / "sft"
    info = train_sft(tiny, llm.load_model(tiny, quantize=False).float(), tok, rows(16), rows(4, "val"), out,
                     r=8, quantized=False)
    assert info["steps"] == info["planned_steps"] == 4 and info["r"] == 8
    losses = [h["eval_loss"] for h in info["history"] if "eval_loss" in h]
    assert losses and losses[0] == pytest.approx(math.log(151936), abs=2.0)   # random model: no double scaling
    assert (out / "adapter" / "adapter_config.json").exists()
    cfg8 = json.loads((out / "adapter" / "adapter_config.json").read_text())
    assert cfg8["r"] == 8 and cfg8["lora_alpha"] == 16
    model = llm.load_model(tiny, quantize=False, adapter=out / "adapter").float()
    preds = llm.predict(model, tok, rows(5, "test"), tiny, log=None)
    assert [p["id"] for p in preds] == [f"test{i}" for i in range(5)]
    assert all(0 <= p["score"] <= 1 and p["pred"] in ("supported", "unsupported") for p in preds)
    # batching must not change scores (left padding + attention mask)
    single = [llm.predict(model, tok, [r], tiny, log=None)[0]["score"] for r in rows(5, "test")]
    assert [p["score"] for p in preds] == pytest.approx(single, abs=1e-4)
    # a finished run is reused; an interrupted one (no train_info.json yet) resumes from its last checkpoint
    assert train_sft(tiny, None, tok, [], [], out)["steps"] == 4
    (out / "train_info.json").unlink()
    again = train_sft(tiny, llm.load_model(tiny, quantize=False).float(), tok, rows(16), rows(4, "val"), out,
                      r=8, quantized=False)
    assert again["resumed_from"] and again["steps"] == 4


def test_time_budget_stops_training(tok, tiny, tmp_path):
    from gjudge.train import train_sft
    cfg = copy.deepcopy(tiny)
    cfg["train"]["max_minutes"] = 0
    info = train_sft(cfg, llm.load_model(tiny, quantize=False).float(), tok, rows(16), rows(4, "val"),
                     tmp_path / "t", quantized=False)
    assert info["stopped_by_time"] and info["steps"] == 1 and (tmp_path / "t" / "adapter").exists()


def test_dpo_pairs(tok):
    from gjudge.train import dpo_pairs
    pairs = dpo_pairs(rows(10), tok, 6, 900)
    assert len(pairs) == 6
    for p in pairs:
        assert p["chosen"].split(".")[0] != p["rejected"].split(".")[0]
        assert p["prompt"].endswith("<|im_start|>assistant\n")
    with pytest.raises(ValueError, match="train split"):
        dpo_pairs(rows(2, "test"), tok, 2, 900)


def test_dpo_trains(tok, tiny, tmp_path):
    from gjudge.train import train_dpo
    info = train_dpo(tiny, llm.load_model(tiny, quantize=False).float(), tok, rows(12), tmp_path / "dpo",
                     quantized=False)
    assert info["pairs"] == 8 and info["steps"] == 2 and (tmp_path / "dpo" / "adapter").exists()
    # TRL must append the end-of-turn token, or the model never learns to stop
    assert info["history"]


def test_qlora_nf4_path_on_cpu(tok, tiny, tmp_path):
    """Real bitsandbytes NF4 load + prepare + LoRA + train + reload. Skipped where bitsandbytes is missing."""
    pytest.importorskip("bitsandbytes")
    from gjudge.train import train_sft
    model = llm.load_model(tiny, quantize=True)
    assert any(type(m).__name__ == "Linear4bit" for m in model.modules())
    info = train_sft(tiny, model, tok, rows(8), rows(4, "val"), tmp_path / "q", quantized=True)
    assert info["steps"] == 2
    again = llm.load_model(tiny, quantize=True, adapter=tmp_path / "q" / "adapter")
    assert llm.predict(again, tok, rows(2, "test"), tiny, log=None)[0]["score"] == pytest.approx(0.5, abs=0.5)
