"""QLoRA training: supervised (SFT) with transformers.Trainer, and optional DPO with TRL."""
import json
import time
from pathlib import Path

import torch
from transformers import (EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from . import llm


class MaxMinutes(TrainerCallback):
    """Stop cleanly (evaluate + save) when the wall-clock budget is used, so a free Colab session is enough."""

    def __init__(self, minutes: float):
        self.minutes, self.hit = minutes, False

    def on_train_begin(self, args, state, control, **kw):
        self.start = time.time()

    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.start > self.minutes * 60:
            self.hit = True
            control.should_evaluate = control.should_save = control.should_training_stop = True


class VerdictTrainer(Trainer):
    """Cross-entropy on the verdict tokens only. With left padding they are the last few positions, so
    logits_to_keep avoids a (batch x 1500 x 151k-vocab) logits tensor that would not fit on a T4."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        n = int((labels != llm.IGNORE).sum(1).max())
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    use_cache=False, logits_to_keep=n + 1)
        logits, target = out.logits[:, :-1].float(), labels[:, -n:]  # position t predicts token t+1
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=llm.IGNORE, reduction="sum")
        # num_items_in_batch counts label tokens over the whole accumulated batch (None in evaluation)
        loss = loss / (num_items_in_batch if num_items_in_batch is not None else (target != llm.IGNORE).sum())
        return (loss, out) if return_outputs else loss


def lora_config(cfg, r: int | None = None):
    from peft import LoraConfig
    lc = cfg["lora"]
    r = r or lc["r"]
    return LoraConfig(r=r, lora_alpha=lc["alpha"] * r // lc["r"],  # keep alpha/r fixed across the rank ablation
                      lora_dropout=lc["dropout"], target_modules=lc["target_modules"], bias="none",
                      task_type="CAUSAL_LM")


def prepare_for_kbit(model):
    from peft import prepare_model_for_kbit_training
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})


def finished(out_dir: Path) -> dict | None:
    """train_info.json is written last, so its presence means the run is complete. Delete the folder to train again."""
    path = out_dir / "train_info.json"
    if not path.exists():
        return None
    print("already trained, reusing", out_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def common_args(cfg, out_dir, quantized: bool) -> dict:
    cuda = torch.cuda.is_available()
    return dict(output_dir=str(out_dir), seed=cfg["seed"], data_seed=cfg["seed"], report_to=[],
                fp16=cuda, use_cpu=not cuda, gradient_checkpointing=quantized,
                gradient_checkpointing_kwargs={"use_reentrant": False} if quantized else None,
                optim="adamw_torch", logging_steps=10, remove_unused_columns=False, disable_tqdm=True)


def train_sft(cfg, model, tok, train_rows, val_rows, out_dir, r: int | None = None, quantized: bool = True) -> dict:
    """Train a LoRA adapter, keep the checkpoint with the lowest validation loss, save it to out_dir/adapter.
    Restarts from the last checkpoint in out_dir if there is one."""
    from peft import get_peft_model
    set_seed(cfg["seed"])
    tc, out_dir = cfg["train"], Path(out_dir)
    if done := finished(out_dir):
        return done
    if quantized:
        model = prepare_for_kbit(model)
    model = get_peft_model(model, lora_config(cfg, r))
    model.config.use_cache = False
    encode = lambda rows: [llm.encode_example(tok, x, cfg["max_tokens"]) for x in rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    steps = -(-len(train_rows) // (tc["batch_size"] * tc["grad_accum"])) * tc["epochs"]
    args = TrainingArguments(
        **common_args(cfg, out_dir, quantized),
        num_train_epochs=tc["epochs"], learning_rate=tc["lr"], lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 20), weight_decay=0.0, max_grad_norm=1.0,
        per_device_train_batch_size=tc["batch_size"], per_device_eval_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["grad_accum"], prediction_loss_only=True,
        eval_strategy="steps", save_strategy="steps", eval_steps=tc["eval_steps"], save_steps=tc["eval_steps"],
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)
    budget = MaxMinutes(tc["max_minutes"])
    trainer = VerdictTrainer(model=model, args=args, train_dataset=encode(train_rows), eval_dataset=encode(val_rows),
                             data_collator=lambda f: llm.collate(f, pad),
                             callbacks=[budget, EarlyStoppingCallback(tc["patience"])])
    resume = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    t = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(out_dir / "adapter"))
    info = {"steps": result.global_step, "planned_steps": steps, "minutes": round((time.time() - t) / 60, 1),
            "stopped_by_time": budget.hit, "resumed_from": resume, "best_eval_loss": trainer.state.best_metric,
            "best_checkpoint": trainer.state.best_model_checkpoint, "r": r or cfg["lora"]["r"],
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "history": trainer.state.log_history}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def dpo_pairs(train_rows, tok, max_pairs: int, max_tokens: int) -> list[dict]:
    """Preference pairs from the TRAIN split: right verdict + short reason over wrong verdict + stock reason.
    The evidence span is the human-marked wrong span from the dataset."""
    ok_reason = "Every claim appears in the passages."
    out = []
    for r in train_rows:
        if r["split"] != "train":
            raise ValueError(f"{r['id']}: DPO pairs may only use the train split")
        if r["label"] == "unsupported":
            if not r.get("evidence"):
                continue
            span = " ".join(r["evidence"].split()[:30])
            chosen = f'unsupported. Reason: the passages do not support "{span}".'
            rejected = f"supported. Reason: {ok_reason}"
        else:
            span = " ".join(r["answer"].split()[:12])
            chosen = f"supported. Reason: {ok_reason}"
            rejected = f'unsupported. Reason: the passages do not support "{span}".'
        p = llm.prompt_text(tok, r)
        if len(tok(p, add_special_tokens=False)["input_ids"]) + 80 <= max_tokens:
            out.append({"prompt": p, "chosen": chosen, "rejected": rejected, "label": r["label"]})
    # balanced and shuffled the same way every time
    import random
    rng = random.Random(0)
    groups = {lab: [x for x in out if x["label"] == lab] for lab in ("supported", "unsupported")}
    n = min(max_pairs // 2, *(len(g) for g in groups.values()))
    pairs = [x for g in groups.values() for x in rng.sample(g, n)]
    rng.shuffle(pairs)
    return [{k: x[k] for k in ("prompt", "chosen", "rejected")} for x in pairs]


def train_dpo(cfg, model, tok, train_rows, out_dir, quantized: bool = True) -> dict:
    """OPTIONAL. DPO with TRL from the BASE model and a fresh adapter (reference = same model, adapter off)."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    set_seed(cfg["seed"])
    dc, out_dir = cfg["dpo"], Path(out_dir)
    if done := finished(out_dir):
        return done
    if quantized:
        model = prepare_for_kbit(model)
    model.config.use_cache = False
    pairs = dpo_pairs(train_rows, tok, dc["max_pairs"], dc["max_tokens"])
    args = DPOConfig(
        **common_args(cfg, out_dir, quantized),
        num_train_epochs=dc["epochs"], learning_rate=dc["lr"], beta=dc["beta"], lr_scheduler_type="cosine",
        per_device_train_batch_size=dc["batch_size"], gradient_accumulation_steps=dc["grad_accum"],
        max_length=dc["max_tokens"], eval_strategy="no", save_strategy="steps", save_steps=50, save_total_limit=1)
    budget = MaxMinutes(dc["max_minutes"])
    trainer = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(pairs), processing_class=tok,
                         peft_config=lora_config(cfg), callbacks=[budget])
    # TRL casts QLoRA adapters to bfloat16; fp16 loss scaling on a T4 cannot unscale bf16 gradients.
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    t = time.time()
    resume = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.model.save_pretrained(str(out_dir / "adapter"))
    info = {"pairs": len(pairs), "steps": result.global_step, "minutes": round((time.time() - t) / 60, 1),
            "stopped_by_time": budget.hit, "history": trainer.state.log_history}
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def merge_adapter(cfg, adapter, out_dir) -> None:
    """Merge the adapter into the float16 base (not into the 4-bit weights, that would add rounding error)."""
    llm.load_model(cfg, quantize=False, adapter=adapter).merge_and_unload().save_pretrained(str(out_dir))
    llm.load_tokenizer(cfg).save_pretrained(str(out_dir))
