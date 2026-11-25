"""Pack what Colab needs into groundedness-judge-colab.zip (code, config, data splits, baseline predictions).

    python scripts/make_colab_zip.py

The zip holds dataset text. Upload it to your own Colab session only; do not publish it.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "groundedness-judge-colab.zip"
REQUIRED = ["data/train.jsonl", "data/val.jsonl", "data/test.jsonl", "data/transfer.jsonl"]
PATTERNS = REQUIRED + ["data/external/*.jsonl", "src/gjudge/*.py", "scripts/run_baselines.py", "configs/*.json", "requirements*.txt",
                       "predictions/*.json*", "docs/data_manifest.json"]


def main(root: Path = ROOT, out: Path = OUT) -> list[str]:
    missing = [f for f in REQUIRED if not (root / f).exists()]
    if missing:
        sys.exit(f"missing {missing}; run scripts/prepare_data.py first")
    names = sorted({p.relative_to(root).as_posix() for pat in PATTERNS for p in root.glob(pat)})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(root / n, n)
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB, {len(names)} files")
    return names


if __name__ == "__main__":
    main()
