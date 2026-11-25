"""Cheap baselines. Each returns one score per row: higher = more likely unsupported."""
import math
import re

import numpy as np

from .metrics import best_threshold
from .schema import context_text, is_unsupported

STOP = set("a an and are as at be been by for from has have he her his i in is it its of on or s she that the "
           "their they this to was were which who will with you your".split())


def words(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", " ", text)  # citation marks are not content
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def lexical_scores(rows) -> list[float]:
    """Share of the answer's content words that do not appear anywhere in the passages or the question."""
    out = []
    for r in rows:
        known = set(words(context_text(r) + " " + r["question"]))
        w = words(r["answer"])
        out.append(sum(x not in known for x in w) / len(w) if w else 0.0)
    return out


def length_features(rows):
    """Nothing but sizes. If this does well, the dataset has a shortcut."""
    return np.array([[math.log1p(len(r["answer"].split())), math.log1p(len(context_text(r).split())),
                      len(r["context"])] for r in rows])


def length_scores(train, rows) -> list[float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(length_features(train), is_unsupported(train))
    return clf.predict_proba(length_features(rows))[:, 1].tolist()


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s*\[\d+\]", "", text)
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text)]
    return [s for s in parts if len(s.split()) >= 3] or [text.strip()]


def windows(row, size: int = 180, stride: int = 120) -> list[str]:
    """Word windows over each passage; the NLI model reads at most 512 tokens."""
    out = []
    for p in row["context"]:
        w = p["text"].split()
        out += [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + stride), stride)]
    return out


class NLI:
    """Off-the-shelf NLI cross-encoder. Score = 1 - min over answer sentences of
    (max over passage windows of P(entailment)): one unsupported sentence is enough."""
    NAME = "cross-encoder/nli-deberta-v3-small"
    REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"

    def __init__(self, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.batch_size = torch, batch_size
        self.tok = AutoTokenizer.from_pretrained(self.NAME, revision=self.REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.NAME, revision=self.REVISION, dtype=torch.float32).eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "laptop CPU"
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    def entailment(self, pairs) -> list[float]:
        out = []
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for k in range(0, len(order), self.batch_size):
            idx = order[k:k + self.batch_size]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                           truncation="only_first", max_length=512, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                p = self.model(**enc).logits.softmax(-1)[:, self.entail].tolist()
            out += list(zip(idx, p))
        return [p for _, p in sorted(out)]

    def score(self, row) -> float:
        sents, wins = sentences(row["answer"]), windows(row)
        p = self.entailment([(w, s) for s in sents for w in wins])
        per_sentence = [max(p[i * len(wins):(i + 1) * len(wins)]) for i in range(len(sents))]
        return 1.0 - min(per_sentence)


def predictions(rows, scores, val_rows, val_scores, latency_s) -> list[dict]:
    """Turn scores into prediction rows. The cut-off comes from validation only."""
    t = best_threshold(is_unsupported(val_rows), val_scores)
    return [{"id": r["id"], "pred": "unsupported" if s >= t else "supported", "score": float(s), "latency_s": latency_s}
            for r, s in zip(rows, scores)]
