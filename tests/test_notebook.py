"""The generated notebook: valid, compiles, in sync with its generator, and (slow, opt-in) runs top to bottom on CPU
with a tiny random model:  GJ_SLOW=1 pytest tests/test_notebook.py"""
import ast
import importlib.util
import json
import os
import zipfile
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_is_in_sync_and_compiles():
    nb = load_script("make_notebook").build()
    nbformat.validate(nb)
    on_disk = nbformat.read(ROOT / "notebooks" / "train_qlora_colab.ipynb", as_version=4)
    assert [c.source for c in on_disk.cells] == [c.source for c in nb.cells], "run scripts/make_notebook.py"
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)


@pytest.mark.skipif(not os.environ.get("GJ_SLOW"), reason="set GJ_SLOW=1 (several minutes)")
def test_notebook_runs_on_cpu(tmp_path, monkeypatch):
    import torch
    from nbclient import NotebookClient
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from gjudge import llm
    from gjudge.schema import write_jsonl

    cfg = json.loads((ROOT / "configs" / "qlora.json").read_text())
    tok = llm.load_tokenizer(cfg)
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, tie_word_embeddings=True, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id)).save_pretrained(tmp_path / "tiny")
    tok.save_pretrained(tmp_path / "tiny")
    cfg.update(base_model=str(tmp_path / "tiny"), revision=None)
    cfg["train"].update(epochs=1, batch_size=2, grad_accum=2, eval_steps=2)
    cfg["dpo"].update(max_pairs=4, batch_size=2, grad_accum=1)

    def rows(split, n):
        return [{"id": f"{split}{i}", "question": "Q?", "answer": f"{split} answer {i}", "split": split, "source": "s",
                 "context": [{"id": "1", "text": f"{split} text {i}"}], "label": ("supported", "unsupported")[i % 2],
                 "group": f"{split}{i}", "evidence": f"span {i}"} for i in range(n)]

    work = tmp_path / "work"
    work.mkdir()
    with zipfile.ZipFile(work / "groundedness-judge-colab.zip", "w") as z:
        for p in (ROOT / "src" / "gjudge").glob("*.py"):
            z.write(p, f"src/gjudge/{p.name}")
        z.writestr("configs/qlora.json", json.dumps(cfg))
        for name, n in (("train", 12), ("val", 4), ("test", 6), ("transfer", 4), ("external/radar", 4)):
            path = tmp_path / "d" / f"{name}.jsonl"
            write_jsonl(path, rows(name.split("/")[-1], n))
            z.write(path, f"data/{name}.jsonl")

    nb = load_script("make_notebook").build()
    for flag in ("RUN_RANK_ABLATION", "RUN_DPO", "RUN_MERGE"):
        nb.cells[1].source = nb.cells[1].source.replace(f"{flag} = False", f"{flag} = True")
    monkeypatch.setenv("GJ_ALLOW_CPU", "1")
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    with zipfile.ZipFile(work / "gj" / "results_bundle.zip") as z:
        names = z.namelist()
    for want in ("docs/results.md", "docs/results.json", "predictions/qwen_sft_r16.jsonl", "predictions/qwen_sft_r8.jsonl",
                 "predictions/qwen_dpo_r16.jsonl", "runs/sft_r16/train_info.json",
                 "runs/sft_r16/adapter/adapter_model.safetensors"):
        assert want in names
    results = json.loads((work / "gj" / "docs" / "results.json").read_text())
    assert results["external:radar.jsonl"]["qwen_sft_r16"] != "not run"
    assert results["test"]["paid_judge"] == "not run"
