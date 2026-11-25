"""Metrics. Positive class = "unsupported" (1). score = how sure the system is that the answer is unsupported."""
import numpy as np


def confusion(y, p) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn)"""
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    return (int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum()))


def classification(y, p) -> dict:
    tp, fp, fn, tn = confusion(y, p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"balanced_accuracy": (rec + spec) / 2, "precision": prec, "recall": rec, "f1": f1,
            "specificity": spec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc(y, s) -> float:
    """Chance that a random unsupported item scores above a random supported one. Ties count half."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return float("nan")
    ranks = rankdata(np.asarray(s, dtype=float))
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(y, s, bins: int = 10) -> list[tuple[float, int, float, float]]:
    """(bin start, count, mean score, observed unsupported rate) for each non-empty equal-width bin."""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    idx = np.minimum((s * bins).astype(int), bins - 1)
    return [(b / bins, int((idx == b).sum()), float(s[idx == b].mean()), float(y[idx == b].mean()))
            for b in range(bins) if (idx == b).any()]


def ece(y, s, bins: int = 10) -> float:
    """Expected calibration error: |mean score - real rate| per bin, weighted by bin size."""
    return sum(n / len(y) * abs(conf - rate) for _, n, conf, rate in reliability(y, s, bins))


def best_threshold(y, s) -> float:
    """Threshold with the highest balanced accuracy (predict unsupported when score >= threshold).
    Use on VALIDATION only. Ties go to the lowest threshold."""
    s = np.asarray(s, dtype=float)
    cands = np.unique(s)
    cands = np.append(cands, np.nextafter(cands[-1], np.inf))
    scores = [classification(y, s >= t)["balanced_accuracy"] for t in cands]
    return float(cands[int(np.argmax(scores))])


def bootstrap_ci(stat, groups, *arrays, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval. Resamples whole groups: six answers about one article are not independent."""
    rng = np.random.default_rng(seed)
    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    members = [np.array(v) for v in by_group.values()]
    arrays = [np.asarray(a) for a in arrays]
    out = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        v = stat(*(a[idx] for a in arrays))
        if v == v:  # skip resamples that lost one class (nan)
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def mcnemar(y, p_a, p_b) -> dict:
    """Exact two-sided McNemar test on the items where exactly one system is right."""
    from scipy.stats import binomtest
    y, p_a, p_b = (np.asarray(v).astype(int) for v in (y, p_a, p_b))
    only_a = int(((p_a == y) & (p_b != y)).sum())
    only_b = int(((p_a != y) & (p_b == y)).sum())
    n = only_a + only_b
    return {"only_a_right": only_a, "only_b_right": only_b,
            "p_value": float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0}
