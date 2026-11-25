"""Generate notebooks/train_qlora_colab.ipynb. The notebook only calls `python -m gjudge.run ...`, which is tested code.

    python scripts/make_notebook.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("md", """# Groundedness judge: QLoRA on a free Colab T4

1. **Runtime > Change runtime type**: **T4 GPU**. Any runtime version works (2026.07 saves about 3 minutes of setup). Save.
2. **Runtime > Run all.**
3. When the upload cell asks, choose `groundedness-judge-colab.zip` (made by `python scripts/make_colab_zip.py`).
4. Keep the tab open. At the end your browser downloads `results_bundle.zip`.

Nothing here needs a login, an API key or GitHub."""),
    ("code", """# Settings. The defaults run the main experiment only.
USE_DRIVE = False          # True: keep checkpoints in your Google Drive, so a dropped session can resume
RUN_RANK_ABLATION = False  # True: also train r=8 (about one more hour)
RUN_DPO = False            # True: also run the optional DPO experiment
RUN_MERGE = False          # True: also save a merged float16 model (about 3 GB, local to the session)
CONFIG = "configs/qlora.json\""""),
    ("code", """# 1. Check the machine. Stops here if it is the wrong runtime.
import os, subprocess, sys
SMOKE = bool(os.environ.get("GJ_ALLOW_CPU"))  # only the local CPU test of this notebook sets this

def sh(cmd):
    \"\"\"Run a command, stream its output, stop the notebook if it fails.\"\"\"
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed with code {p.returncode}: {cmd}")

PY = f'"{sys.executable}"'
if not SMOKE:
    if subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode != 0:
        raise RuntimeError("No GPU. Runtime > Change runtime type > T4 GPU, then Run all again.")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# The pinned packages were checked on Python 3.12. A newer runtime gets its own 3.12 environment; every step below runs in it.
OWN_ENV = not SMOKE and sys.version_info[:2] != (3, 12)
if OWN_ENV:
    PY = '"/content/gjenv/bin/python"'
print(sys.version, "| steps run on", PY)"""),
    ("code", """# 2. Upload the project zip (or find one already here) and unpack it.
import glob, zipfile
try:
    from google.colab import files
except ImportError:
    files = None  # not Colab (Kaggle, local test): put the zip next to the notebook
if os.path.basename(os.getcwd()) != "gj":
    zips = glob.glob("groundedness-judge-colab*.zip")
    if not zips and files:
        zips = list(files.upload())
    if not zips:
        raise RuntimeError("No zip uploaded.")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall("gj")
    os.chdir("gj")
os.environ["PYTHONPATH"] = "src"
os.environ["PYTHONUTF8"] = "1"
for f in ("data/train.jsonl", "data/val.jsonl", "data/test.jsonl", "data/transfer.jsonl", CONFIG):
    if not os.path.exists(f):
        raise RuntimeError(f"{f} is not in the zip. Run scripts/prepare_data.py, then scripts/make_colab_zip.py again.")
print(sorted(os.listdir(".")), "external sets:", glob.glob("data/external/*.jsonl"))"""),
    ("code", """# 3. Install the pinned packages (about 2 minutes). torch is the one Colab ships.
if OWN_ENV:
    sh(f'"{sys.executable}" -m pip install -q uv')
    sh(f'"{sys.executable}" -m uv venv -q --clear --python 3.12 /content/gjenv')
    sh(f'"{sys.executable}" -m uv pip install -q --python {PY} --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0')
    sh(f'"{sys.executable}" -m uv pip install -q --python {PY} -r requirements-colab.txt sentencepiece==0.2.2 protobuf==5.29.6')
    sh(f'"{sys.executable}" -m uv pip freeze --python {PY} > pip-freeze.txt')
elif not SMOKE:
    sh(f"{PY} -m pip install -q -r requirements-colab.txt")
    sh(f"{PY} -m pip freeze > pip-freeze.txt")
if not SMOKE:
    sh(f'{PY} -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"')
sh(f'{PY} -c "import torch, transformers, peft, trl, bitsandbytes as b; '
   f'print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, b.__version__)"')"""),
    ("code", """# 4. Where checkpoints go.
RUNS = "runs"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS = "/content/drive/MyDrive/groundedness-judge-runs"
os.makedirs(RUNS, exist_ok=True)
print("runs in", RUNS)"""),
    ("code", """# 5. The untrained model first: zero-shot and 4-shot (estimate 20 to 30 minutes together).
sh(f"{PY} -m gjudge.run predict --config {CONFIG} --system qwen_zero_shot")
sh(f"{PY} -m gjudge.run predict --config {CONFIG} --system qwen_few_shot --shots 4")"""),
    ("code", """# 5b. The off-the-shelf NLI baseline on all rows (estimate 5 to 10 minutes on the GPU).
if not SMOKE:
    sh(f"{PY} scripts/run_baselines.py --nli")"""),
    ("code", """# 6. QLoRA training, r=16. Hard limit 70 minutes; resumes if a checkpoint exists in RUNS.
sh(f'{PY} -m gjudge.run train --config {CONFIG} --r 16 --out "{RUNS}/sft_r16"')
sh(f'{PY} -m gjudge.run predict --config {CONFIG} --system qwen_sft_r16 --adapter "{RUNS}/sft_r16/adapter"')"""),
    ("code", """# 7. OPTIONAL: rank ablation, r=8 with the same alpha/r ratio. Off by default.
if RUN_RANK_ABLATION:
    sh(f'{PY} -m gjudge.run train --config {CONFIG} --r 8 --out "{RUNS}/sft_r8"')
    sh(f'{PY} -m gjudge.run predict --config {CONFIG} --system qwen_sft_r8 --adapter "{RUNS}/sft_r8/adapter"')"""),
    ("code", """# 8. OPTIONAL: DPO from the base model on preference pairs made from the train split. Off by default.
if RUN_DPO:
    sh(f'{PY} -m gjudge.run dpo --config {CONFIG} --out "{RUNS}/dpo_r16"')
    sh(f'{PY} -m gjudge.run predict --config {CONFIG} --system qwen_dpo_r16 --adapter "{RUNS}/dpo_r16/adapter"')"""),
    ("code", """# 9. OPTIONAL: merged float16 model (not needed for the results). Off by default.
if RUN_MERGE:
    sh(f'{PY} -m gjudge.run merge --config {CONFIG} --adapter "{RUNS}/sft_r16/adapter" --out merged')"""),
    ("code", """# 10. Score everything: docs/results.md and docs/results.json.
sh(f"{PY} -m gjudge.evaluate")
print(open("docs/results.md", encoding="utf-8").read())"""),
    ("code", """# 11. Look at the loss curves (text, no plotting library needed).
import json
for info_path in sorted(glob.glob(f"{RUNS}/*/train_info.json")):
    info = json.load(open(info_path))
    print(info_path, {k: v for k, v in info.items() if k not in ("history",)})
    for h in info["history"]:
        if "eval_loss" in h:
            print("  step", h["step"], "validation loss", round(h["eval_loss"], 4))
        elif "loss" in h:
            print("  step", h["step"], "train loss", round(float(h["loss"]), 4))"""),
    ("code", """# 12. Pack and download: predictions, results, loss history, LoRA adapters.
with zipfile.ZipFile("results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pattern in ("predictions/*", "docs/results.*", "pip-freeze.txt"):
        for f in glob.glob(pattern):
            z.write(f)
    for f in glob.glob(f"{RUNS}/*/train_info.json") + glob.glob(f"{RUNS}/*/adapter/*"):
        z.write(f, os.path.join("runs", os.path.relpath(f, RUNS)))
print(os.path.getsize("results_bundle.zip") // 1_000_000, "MB")
if files:
    files.download("results_bundle.zip")
    files.download("docs/results.json")"""),
]


def build():
    cells = [new_markdown_cell(t) if kind == "md" else new_code_cell(t) for kind, t in CELLS]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"  # stable ids, so regenerating gives no diff
    return new_notebook(cells=cells, metadata={
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


if __name__ == "__main__":
    out = ROOT / "notebooks" / "train_qlora_colab.ipynb"
    nb = build()
    nbformat.validate(nb)
    nbformat.write(nb, out)
    print("wrote", out)
