"""Fast tests: no model, no network."""
import json
import math

import numpy as np
import pytest

from gjudge import baselines, data, evaluate, leakage, metrics as M, prompt
from gjudge.schema import load_jsonl, validate, write_jsonl


def row(id="a", answer="Paris is big.", text="Paris is a big city in France.", label="supported", **kw):
    return {"id": id, "question": "Q?", "context": [{"id": "1", "text": text}], "answer": answer, "label": label,
            "source": "s", "split": "test", **kw}


# --- contract -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("broken", [{"label": "maybe"}, {"context": []}, {"answer": " "}, {"id": 3},
                                    {"context": [{"id": 1, "text": "x"}]}])
def test_contract_rejects(broken):
    with pytest.raises(ValueError):
        validate({**row(), **broken})


def test_load_rejects_duplicate_ids(tmp_path):
    write_jsonl(tmp_path / "x.jsonl", [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(tmp_path / "x.jsonl")


# --- prompt and parsing ---------------------------------------------------------------------------------------
def test_prompt_contains_everything_but_never_the_label():
    for label in ("supported", "unsupported"):
        r = row(label=label, answer="ANS", text="CTX", evidence="SECRET_SPAN")
        m = prompt.messages(r)
        assert [x["role"] for x in m] == ["system", "user"]
        assert "[1] CTX" in m[1]["content"] and "ANS" in m[1]["content"] and "SECRET_SPAN" not in str(m)
        assert m == prompt.messages({**r, "label": "supported"})


def test_few_shot_turns():
    m = prompt.messages(row(), [row(label="unsupported"), row()])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert m[2]["content"] == "unsupported"


@pytest.mark.parametrize("text,want", [
    ("supported", "supported"), ("Unsupported.", "unsupported"), ("The answer is NOT supported", "unsupported"),
    ("unsupported. Reason: not all supported", "unsupported"), ("Supported\nbecause", "supported"),
    ("yes", None), ("", None), ("unsupportedly", None), ("not  supported", "unsupported")])
def test_parse_verdict(text, want):
    assert prompt.parse_verdict(text) == want


# --- metrics, checked by hand ---------------------------------------------------------------------------------
Y = [1, 1, 1, 0, 0]
P = [1, 1, 0, 1, 0]  # tp=2 fn=1 fp=1 tn=1


def test_classification_by_hand():
    c = M.classification(Y, P)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["precision"] == pytest.approx(2 / 3) and c["recall"] == pytest.approx(2 / 3)
    assert c["f1"] == pytest.approx(2 / 3) and c["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_auroc_by_hand():
    # pairs (pos, neg): .9>.5 yes, .9>.1 yes, .5=.5 half, .5>.1 yes -> 3.5 of 4
    assert M.auroc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]) == pytest.approx(3.5 / 4)
    assert math.isnan(M.auroc([1, 1], [0.1, 0.2]))


def test_ece_by_hand():
    # bin 0: scores .05,.05 rate 0.5 -> gap .45; bin 9: .95,1.0 mean .975 rate 1 -> gap .025; each weight 0.5
    assert M.ece([0, 1, 1, 1], [0.05, 0.05, 0.95, 1.0]) == pytest.approx(0.5 * 0.45 + 0.5 * 0.025)


def test_best_threshold():
    assert M.best_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]) == pytest.approx(0.6)
    assert M.best_threshold([1, 1, 0, 0], [0.1, 0.1, 0.1, 0.1]) == pytest.approx(0.1)


def test_mcnemar_by_hand():
    y = [1] * 10
    a, b = [1] * 10, [0] * 6 + [1] * 4          # 6 discordant, all won by A: p = 2 * 0.5**6
    m = M.mcnemar(y, a, b)
    assert (m["only_a_right"], m["only_b_right"]) == (6, 0) and m["p_value"] == pytest.approx(2 * 0.5 ** 6)
    assert M.mcnemar(y, a, a)["p_value"] == 1.0


def test_bootstrap_resamples_groups():
    y = np.array([1, 0] * 50)
    p = np.array([1, 0] * 25 + [0, 1] * 25)
    f = lambda a, b: float((a == b).mean())
    lo, hi = M.bootstrap_ci(f, list(range(100)), y, p, n=300)
    assert lo < 0.5 < hi and hi - lo < 0.25
    # one big right group and one big wrong group: only two units, so the interval must be very wide
    lo2, hi2 = M.bootstrap_ci(f, [0] * 50 + [1] * 50, y, p, n=300)
    assert lo2 == 0.0 and hi2 == 1.0


# --- leakage --------------------------------------------------------------------------------------------------
LONG = "the quick brown fox jumps over the lazy dog while the small cat sleeps near the warm stove all day"


def test_near_duplicate_found_and_unrelated_not():
    assert leakage.near_duplicates([LONG], [LONG.upper() + "!!"])[0][2] == 1.0
    assert leakage.near_duplicates([" ".join(LONG.split()[:10])], ["intro text here. " + LONG])  # contained
    assert leakage.near_duplicates([LONG], ["completely different words about tax law and shipping rates"]) == []


def test_leaky_indices_reasons():
    test = [row("t1", text=LONG, group="g1")]
    train = [row("a", text="other words entirely about nothing much at all", group="g1"),
             row("b", text=LONG, answer=" paris is BIG", group="g2"),
             row("c", text=LONG + " and more words after it", answer="x", group="g3"),
             row("d", text="something unrelated about the price of steel in 1970", group="g4")]
    assert leakage.leaky_indices(train, test) == {0: "same_group", 1: "exact_row", 2: "near_duplicate_passage"}
    assert leakage.overlap_report(train[3:], test) == dict.fromkeys(leakage.overlap_report([], []), 0)
    assert leakage.exact_duplicates([train[1], {**train[1], "id": "z"}]) == {1}


# --- data -----------------------------------------------------------------------------------------------------
def test_ragtruth_rows():
    src = [{"source_id": "1", "task_type": "Summary", "source_info": "text"},
           {"source_id": "2", "task_type": "QA", "source_info":
               {"question": "q", "passages": "passage 1:aa\n\npassage 2:bb\n\n"}}]
    resp = [{"id": "9", "source_id": "1", "quality": "good", "labels": [], "split": "train", "response": "r"},
            {"id": "8", "source_id": "1", "quality": "truncated", "labels": [], "split": "train", "response": "r"},
            {"id": "7", "source_id": "2", "quality": "good", "split": "test", "response": "r",
             "labels": [{"text": "bad", "implicit_true": True}]}]
    s = data.ragtruth_rows(resp, src, "Summary")
    assert [x["id"] for x in s] == ["ragtruth-summary-9"] and s[0]["label"] == "supported"
    q = data.ragtruth_rows(resp, src, "QA")[0]
    assert q["label"] == "unsupported" and q["hard"] and q["context"] == [{"id": "1", "text": "aa"}, {"id": "2", "text": "bb"}]


def test_hagrid_groups_share_article_and_unlabelled_dropped():
    q = lambda i, doc: {"query_id": i, "query": "q", "quotes": [{"docid": doc, "text": "t"}],
                        "answers": [{"answer": "a", "attributable": 0, "sentences": [{"text": "a", "attributable": 0}]},
                                    {"answer": "b", "sentences": []}]}
    rows = data.hagrid_rows([q(1, "7#1"), q(2, "7#9"), q(3, "8#0")])
    assert len(rows) == 3 and rows[0]["group"] == rows[1]["group"] != rows[2]["group"]
    assert rows[0]["evidence"] == "a"


def test_split_is_by_document_and_stable():
    assert data.stable_fraction("doc", 1) == data.stable_fraction("doc", 1) != data.stable_fraction("doc", 2)
    xs = [data.stable_fraction(str(i), 1) for i in range(2000)]
    assert 0.47 < np.mean(xs) < 0.53


def test_balance():
    rows = [row(str(i), label="supported") for i in range(9)] + [row(f"u{i}", label="unsupported") for i in range(4)]
    b = data.balance(rows, 1)
    assert len(b) == 8 and sum(r["label"] == "supported" for r in b) == 4
    assert b == data.balance(list(reversed(rows)), 1) and len(data.balance(rows, 1, 2)) == 4


# --- baselines and evaluation ---------------------------------------------------------------------------------
def test_lexical_and_windows():
    s = baselines.lexical_scores([row(answer="Paris is big [1]."), row(answer="Zebras eat Paris")])
    assert s[0] == 0.0 and s[1] == pytest.approx(2 / 3)
    r = row(text=" ".join(str(i) for i in range(500)))
    seen = {w for win in baselines.windows(r) for w in win.split()}
    assert len(seen) == 500                      # no evidence lost
    assert len(baselines.windows(row())) == 1
    assert baselines.sentences("One two three [1]. Four five six.") == ["One two three.", "Four five six."]


def test_evaluate_end_to_end(tmp_path):
    D = tmp_path / "data"
    for split in ("val", "test", "transfer", "train"):
        write_jsonl(D / f"{split}.jsonl", [
            row(f"{split}{i}", label=("supported", "unsupported")[i % 2], group=f"{split}-g{i // 2}",
                text=f"{split} passage number {i} " * 3) for i in range(20)])
    ext = tmp_path / "radar.jsonl"
    write_jsonl(ext, [row(f"e{i}", label=("supported", "unsupported")[i % 2], text=f"ext {i}") for i in range(6)])
    (tmp_path / "docs").mkdir()
    ids = [f"{s}{i}" for s in ("val", "test", "transfer") for i in range(20)]
    # scores separate the classes perfectly at 0.3, not at 0.5: only a validation cut-off gets this right
    preds = [{"id": x, "pred": "supported", "score": 0.4 if int(x.lstrip("valtesrnf")) % 2 else 0.2, "latency_s": 0.01}
             for x in ids]
    write_jsonl(tmp_path / "predictions" / "lexical_overlap.jsonl", preds)
    (tmp_path / "predictions" / "lexical_overlap.meta.json").write_text(json.dumps({"title": "Lexical overlap"}))
    evaluate.main(["--root", str(tmp_path), "--external", str(ext)])
    text = (tmp_path / "docs" / "results.md").read_text()
    assert "| Lexical overlap | 1.000 (1.000-1.000)" in text
    assert "| Paid judge model | not run" in text and "radar.jsonl" in text
    # a system that skips rows must not be scored silently on the full set
    write_jsonl(tmp_path / "predictions" / "lexical_overlap.jsonl", preds[:-1])
    evaluate.main(["--root", str(tmp_path)])
    transfer = (tmp_path / "docs" / "results.md").read_text().split("## Transfer test")[1].split("##")[0]
    assert "| Lexical overlap | sample only" in transfer
