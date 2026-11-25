# Groundedness judge

Can a small open model be trained to check whether a RAG answer is really supported by its passages?
This repo fine-tunes Qwen2.5-1.5B-Instruct with QLoRA for that one decision and compares it, fairly, with cheap baselines.
A negative result is a valid result. No number is written here that a script did not produce.

- **Input**: question, passages, answer. **Output**: `supported` or `unsupported`, plus P(unsupported).
- **Train**: 3,580 rows. **Validation**: 490. **Test**: 866. **Transfer test** (other task, never trained on): 320.
- **External test**: a small human-labelled set from a real RAG app, loaded by path, never trained on.

## Status

| Step | State |
|---|---|
| Data, leakage checks, dataset card (`docs/data.md`) | done |
| CPU baselines: majority, length probe, word overlap | done, see `docs/results.md` |
| NLI baseline | run on every row on the T4 |
| Training code, tested on CPU with a tiny random model, including real 4-bit NF4 loading | done |
| Full notebook run on CPU with a tiny model (`GJ_SLOW=1 pytest tests/test_notebook.py`) | passed end to end (2026-11-25, 7.5 minutes) |
| Zero-shot, few-shot, QLoRA runs | zero-shot, 4-shot and QLoRA r=16 done on a free T4 (2026-11-25); r=8 and DPO not run |

## Results so far

All numbers are in [docs/results.md](docs/results.md), written by `python -m gjudge.evaluate`. Systems that were not run say "not run".

| System | Main test (866): balanced accuracy | AUROC | Transfer (320): balanced accuracy | AUROC | Radar, human labels (52): balanced accuracy | AUROC |
|---|---|---|---|---|---|---|
| Length only (shortcut probe) | 0.559 (0.522-0.591) | 0.607 | 0.512 (0.503-0.528) | 0.702 | 0.515 (0.443-0.599) | 0.625 |
| Lexical overlap | 0.624 (0.595-0.656) | 0.644 | 0.688 (0.635-0.741) | 0.765 | 0.736 (0.608-0.856) | 0.768 |
| NLI DeBERTa-v3-small | 0.583 (0.551-0.615) | 0.646 | 0.641 (0.602-0.680) | 0.749 | 0.721 (0.592-0.840) | 0.840 |
| Qwen2.5-1.5B, zero-shot | 0.579 (0.548-0.610) | 0.635 | 0.550 (0.489-0.608) | 0.570 | 0.634 (0.524-0.750) | 0.836 |
| Qwen2.5-1.5B, 4-shot | 0.596 (0.563-0.629) | 0.646 | 0.541 (0.475-0.603) | 0.568 | 0.643 (0.548-0.740) | 0.879 |
| **Qwen2.5-1.5B + QLoRA r=16** | **0.758 (0.728-0.787)** | **0.847** | 0.634 (0.593-0.673) | **0.860** | 0.643 (0.545-0.739) | **0.905** |

One run on a free Colab T4 (2026-11-25), one seed. Training stopped at its 70-minute limit after 207 of 448 planned steps (less than one
pass over the data); validation loss was still about flat (0.267, 0.264, 0.270), with no sign of overfitting. r=8, DPO and a paid judge were not run.

What it shows:

- **Fine-tuning worked on its own kind of data.** 0.579 to 0.758 on the main test, better than every baseline (McNemar p < 0.0001 against each).
  Untuned, the model answered "supported" 99.9% of the time, so its 0.579 already needs a cut-off picked on validation.
- **The ranking carries over, the cut-off does not.** On both outside sets it has the best AUROC (0.860, 0.905), but with the
  cut-off frozen on validation it calls too much unsupported, and plain word overlap beats it on balanced accuracy. Using the generated word
  instead of the tuned cut-off gives 0.756 on transfer and 0.738 on Radar. An exploratory check, decided after seeing this: picking the
  cut-off on half of the outside set and scoring the other half gives about 0.78 (transfer) and 0.81 (Radar, 52 rows, wide).
- **Recall is high, trust is not.** It flagged all 31 unsupported Radar answers, and also 11 of the 21 supported ones.
- **So it is not used in Radar.** It would need a cut-off calibrated on Radar's own labels, and more of them. The hosted judge used there agreed
  with the same person on 27 of 32 real answers.

95% intervals in brackets, bootstrap over source documents. AUROC intervals are in docs/results.md.

## Run it

### A. On your own machine (no GPU needed)

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-local.txt
.venv/Scripts/python scripts/prepare_data.py       # about 300 MB of downloads, 5 minutes
.venv/Scripts/python scripts/run_baselines.py      # seconds
.venv/Scripts/python -m pytest                     # about 2 minutes
.venv/Scripts/python scripts/make_colab_zip.py     # writes groundedness-judge-colab.zip (about 3 MB)
```

On Windows set `PYTHONUTF8=1`. On Linux or macOS use `.venv/bin/python`.
To run `gjudge` modules directly, set `PYTHONPATH=src`.

### B. On Google Colab (free T4), step by step

You need a Google account. No payment, no Hugging Face login, no API key.

1. Open <https://colab.research.google.com>. **File > Upload notebook** and choose `notebooks/train_qlora_colab.ipynb`.
2. **Runtime > Change runtime type**. Hardware accelerator: **T4 GPU**. Any runtime version works (a newer one builds its own Python 3.12 environment, about 3 minutes). Click **Save**.
   The notebook stops with a clear message if either is wrong.
3. (Optional) In the first code cell set `USE_DRIVE = True`. Then checkpoints go to your Drive and a lost session can continue.
   A Google permission window opens when the Drive cell runs.
4. **Runtime > Run all**. If Colab warns that the notebook was not made by Google, choose **Run anyway**.
5. The second code cell shows a **Choose Files** button. Choose `groundedness-judge-colab.zip`.
6. Wait. Keep the browser tab open and the laptop awake. Estimates (not measured on a T4 yet):

   | Step | Estimate |
   |---|---|
   | Install and model download | 5 min |
   | Zero-shot and 4-shot | 20 to 30 min |
   | NLI on GPU | 5 to 10 min |
   | Training r=16 | up to 70 min (hard limit) |
   | Tuned predictions and scoring | 10 to 15 min |

   Total about 2 hours in the worst case, normally less. Free sessions can end early: if that happens, set `USE_DRIVE = True` and run again.
7. At the end the browser downloads `results_bundle.zip` and `results.json`. If it does not, open the Files panel (folder
   icon on the left), folder `gj`, right click `results_bundle.zip`, Download.
8. On your machine, unzip it in the repo root (it fills `predictions/` and `runs/`), then run
   `python -m gjudge.evaluate` to rebuild `docs/results.md`.
9. Read `docs/results.md` top to bottom before you quote anything.

Optional parts, each in a new session so the limit is not hit (`USE_DRIVE = True` is best): `RUN_RANK_ABLATION` (r=8, about 80 more minutes),
`RUN_DPO` (about 60), `RUN_MERGE` (a few minutes). In a later session the finished r=16 run is found in Drive and not trained again;
to keep the earlier prediction files, unzip the results bundle into the repo and make a new zip first.

### C. Add the external (Radar) set

Put the JSONL in `data/external/` **before** making the Colab zip, run `scripts/run_baselines.py` again, then step B. Same row format as
`docs/data.md`. To score only: `python -m gjudge.evaluate --external path/to/file.jsonl` (needs predictions for its ids).

### D. Add a paid judge later

Write `predictions/paid_judge.jsonl`, one line per row of val, test and transfer:
`{"id": "...", "pred": "supported"|"unsupported", "score": 0.0-1.0 or null, "latency_s": 1.2}`. `evaluate` picks it up.

## Files

| Path | What |
|---|---|
| `src/gjudge/schema.py`, `prompt.py` | data contract, prompt, verdict parser |
| `src/gjudge/data.py`, `leakage.py`, `scripts/prepare_data.py` | build the splits and check them |
| `src/gjudge/baselines.py`, `scripts/run_baselines.py` | majority, length probe, word overlap, NLI |
| `src/gjudge/llm.py`, `train.py`, `run.py` | model loading, prediction, SFT and DPO, command line |
| `src/gjudge/metrics.py`, `evaluate.py` | metrics, confidence intervals, `docs/results.md` |
| `scripts/make_notebook.py` | generates the notebook; edit this, not the notebook |
| `docs/data.md`, `docs/train_info_sft_r16.json` | dataset card and licences; loss history of the GPU run |

Not in this repository: the trained weights, the dataset text (the scripts download it from its sources) and the Radar answers.

## Limits

- Balanced, public, English data; no rows over 1,536 tokens; one base model; one seed.
- Labels have noise (RAGTruth annotators agreed on 91.8% of answers), so 100% is not reachable.
- HAGRID has a length shortcut. Read the per-source table.
- The GPU path has run once, on a T4. Batch size and limits are in `configs/qlora.json`.
- The Radar set is 52 rows from one labeller, and 10 of them are mismatched on purpose (easy).
- Trained weights: read the open licence points in `docs/data.md` before publishing them.
