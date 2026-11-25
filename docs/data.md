# Dataset card

Built by `scripts/prepare_data.py` (seed 17, pinned source commits). Exact counts, file hashes and the overlap
report are in `docs/data_manifest.json`. No dataset text is committed: `data/` is gitignored.

## One row

```
{"id", "question", "context": [{"id", "text"}], "answer", "label": "supported"|"unsupported", "source", "split"}
```

Two extra keys help inside this repo and may be missing elsewhere: `group` (the source document, used for
splitting and for the bootstrap) and `evidence` (the human-marked wrong span, used only for DPO pairs).
`hard` is true when every marked span is "true in the world but not in the source" (RAGTruth `implicit_true`).

## Sources and licences (each read from its own repo or card on 2026-11-24)

| Source | Where the licence was read | Licence | Used for |
|---|---|---|---|
| RAGTruth, Summary task | github.com/ParticleMedia/RAGTruth, file `LICENSE` | MIT | train, val, test |
| CNN/DailyMail (upstream of most RAGTruth summaries) | huggingface.co/datasets/abisee/cnn_dailymail card | card says Apache-2.0 | upstream of the above |
| HAGRID | huggingface.co/datasets/miracl/hagrid card and github.com/project-miracl/hagrid README | Apache-2.0 | train, val, test |
| VitaminC (real revisions) | github.com/TalSchuster/VitaminC, file `DATA_LICENSE` | Wikipedia terms, else CC BY-SA 3.0 | train, val, test |
| RAGTruth, QA task | same MIT file; passages come from MS MARCO | MIT for RAGTruth; MS MARCO page: "non-commercial research purposes only ... without extending any license" | **evaluation only** |
| Qwen2.5-1.5B-Instruct and 0.5B-Instruct | `LICENSE` in each Hugging Face repo | Apache-2.0 | base model / smoke test |
| cross-encoder/nli-deberta-v3-small | Hugging Face card | Apache-2.0 | baseline |

Looked at and not used:

| Candidate | Why not |
|---|---|
| RAGTruth Data2txt | Yelp business records as JSON: not RAG text, and Yelp data has its own terms |
| RAGTruth QA for training | MS MARCO terms do not clearly allow public release of trained weights |
| LLM-AggreFact | card: CC BY-ND 4.0, gated, "should not be used in pretraining or fine-tuning any NLP models" |
| HaluBench | card: CC BY-NC 2.0 |
| ANLI | repo `LICENSE`: CC BY-NC 4.0 |
| HaluEval QA | MIT, but the negatives are written by ChatGPT and are longer than the right answer in 95.8% of 10,000 pairs; "5 words or more = hallucinated" already scores 94.5% (measured here). A model would learn length |
| FEVER, SciFact | licences are fine (Wikipedia / CC BY-SA 3.0; CC BY 4.0 claims with ODC-By abstracts) but VitaminC already covers short claims |
| RAGBench | CC BY 4.0, but labels come from GPT-4o, not people |
| Llama-3.2-1B | gated (login needed) with its own community licence; Qwen is Apache-2.0 |
| SmolLM2-1.7B | Apache-2.0, a fine choice too; one base model is enough here |

Not legal advice. Open points to decide before publishing **weights** (results are fine either way): VitaminC and
HAGRID passages are Wikipedia text (share-alike, so credit them), and RAGTruth "Recent News" articles come from a
news site that the paper does not name; the MIT file covers the RAGTruth release, not necessarily each article.
To train without VitaminC, set its train cap to 0 in `CAPS`.

## Who made the labels

- **RAGTruth**: real answers from six LLMs; paid annotators marked wrong spans, two per answer plus a third on
  disagreement (91.8% agreement per answer, arXiv 2401.00396). Any span makes the answer unsupported, including
  `implicit_true` spans: the judge checks grounding, not truth.
- **HAGRID**: GPT-3.5 answers with citations; people judged if the answer is attributable to the quotes
  (arXiv 2307.16883). Answers without a label were dropped. Known noise: a right claim with a wrong citation number can
  be marked not attributable.
- **VitaminC**: claims written for pairs of Wikipedia revisions, so nearly the same sentence is supported by one
  version and not by the other. REFUTES and NOT ENOUGH INFO both become unsupported. Plain word overlap fails by design.

## Splits

Split by source document before any filter: RAGTruth keeps its official test articles (validation is cut from official
train articles); HAGRID is split by query groups, where queries that cite the same Wikipedia article are joined; VitaminC keeps its
official train/dev/test, at most one supported and one unsupported row per page. Then each source is balanced 50/50.

| Split | Rows | Documents | ragtruth_summary | hagrid | vitaminc | ragtruth_qa | Mean tokens | Max |
|---|---|---|---|---|---|---|---|---|
| train | 3,580 | 1,837 | 2,078 | 502 | 1,000 | - | 624 | 1,536 |
| val | 490 | 304 | 250 | 120 | 120 | - | 638 | 1,536 |
| test | 866 | 544 | 332 | 374 | 160 | - | 632 | 1,530 |
| transfer (evaluation only) | 320 | 140 | - | - | - | 320 | 592 | 1,049 |

`test` and `transfer` are used once per system, after every choice is fixed. Cut-offs come from `val`.
`data/external/*.jsonl` (the Radar set) is read only by the prediction and evaluation code; `load_train` in
`src/gjudge/run.py` refuses to start if an external id is found in the training file.

## Long contexts

Nothing is truncated, so no evidence can be cut away. A row is kept only if its whole prompt fits in 1,536 tokens
(Qwen tokenizer). Cost of this rule: the official RAGTruth summary test goes from 900 to 787 rows (and from
204 to 166 unsupported), HAGRID test from 690 to 662; RAGTruth QA loses nothing. So results say nothing about very long documents.
The NLI baseline reads at most 512 tokens, so it gets overlapping 180-word windows that cover everything.

## Leakage checks (all in `src/gjudge/leakage.py`)

1. Exact duplicates (answer + context after normalising) inside a split are dropped.
2. A train or val row is dropped if it shares a document with, equals, or has a passage near-identical to a row in a cleaner
   split (order: test and transfer, then val, then train). Near-identical = at least half of the 5-word
   sequences of the smaller passage also appear in the other one.
3. After all that, the script recounts train-val, train-test, train-transfer, val-test, val-transfer
   and fails if any count is not zero. All were zero.

Dropped because of rules 1 and 2: train 9 exact duplicates and 105 near-duplicate rows (42 RAGTruth, 10 HAGRID, 53 VitaminC);
val 1 and 67 (65 VitaminC, 2 HAGRID). The 42 RAGTruth rows match a test or a
validation article. Worth knowing: in the official RAGTruth summary split, 24 train rows (4 articles) are near-copies of test articles.

## Known weak points

- Balanced classes: precision and F1 here are not what you would see on live traffic.
- HAGRID has a length shortcut: answer length alone reaches 0.655 balanced accuracy there (0.476 on RAGTruth
  summaries, 0.506 on VitaminC). That is why `length_only` is in the results table. Beat it, per source, or the result means little.
- Training has no RAG QA with human labels except the 502 HAGRID rows. The transfer test shows if that matters.
- Public since 2021 to 2024: the base model may have seen parts of these sets during pretraining. Cannot be checked.
