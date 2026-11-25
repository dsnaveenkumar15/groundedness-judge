# Results

Written by `python -m gjudge.evaluate`. Do not edit by hand.

Positive class = unsupported. Cut-offs were chosen on validation only. 95% CIs: bootstrap over source documents, 1000 resamples. Both test sets are balanced, so precision and F1 do not carry over to live traffic where unsupported answers are rarer; balanced accuracy, recall and AUROC do.

Latency is batched throughput on the named device, so compare it only within one device.

## Main test

866 rows from 544 source documents: 433 supported, 433 unsupported.

| System | Balanced accuracy (95% CI) | Precision | Recall | F1 (95% CI) | AUROC (95% CI) | ECE | ms per item (device) | Size MB |
|---|---|---|---|---|---|---|---|---|
| Majority class (always unsupported) | 0.500 (0.500-0.500) | 0.500 | 1.000 | 0.667 (0.638-0.697) | 0.500 (0.500-0.500) | - | 0.0 (-) | 0 |
| Length only (shortcut probe) | 0.559 (0.522-0.591) | 0.550 | 0.654 | 0.597 (0.556-0.634) | 0.607 (0.563-0.647) | 0.047 | 0.0 (laptop CPU) | 0 |
| Lexical overlap | 0.624 (0.595-0.656) | 0.659 | 0.513 | 0.577 (0.536-0.616) | 0.644 (0.608-0.680) | - | 0.2 (laptop CPU) | 0 |
| NLI DeBERTa-v3-small (off the shelf) | 0.583 (0.551-0.615) | 0.565 | 0.718 | 0.633 (0.595-0.665) | 0.646 (0.606-0.681) | 0.368 | 159.4 (Tesla T4) | 568 |
| Qwen2.5-1.5B base, zero-shot | 0.579 (0.548-0.610) | 0.577 | 0.589 | 0.583 (0.543-0.621) | 0.635 (0.599-0.672) | 0.487 | 116.7 (Tesla T4) | 1122 |
| Qwen2.5-1.5B base, 4-shot | 0.596 (0.563-0.629) | 0.596 | 0.596 | 0.596 (0.556-0.636) | 0.646 (0.610-0.684) | 0.448 | 312.5 (Tesla T4) | 1122 |
| Qwen2.5-1.5B + QLoRA SFT r=16 | 0.758 (0.728-0.787) | 0.735 | 0.806 | 0.769 (0.736-0.799) | 0.847 (0.824-0.872) | 0.061 | 202.8 (Tesla T4) | 1196 |
| Qwen2.5-1.5B + QLoRA SFT r=8 (ablation) | not run | | | | | | | |
| Qwen2.5-1.5B + QLoRA DPO r=16 (optional) | not run | | | | | | | |
| Paid judge model | not run | | | | | | | |

### Balanced accuracy by source

| System | hagrid | ragtruth_summary | vitaminc |
|---|---|---|---|
| Majority class (always unsupported) | 0.500 (0.500-0.500) | 0.500 (0.500-0.500) | 0.500 (0.500-0.500) |
| Length only (shortcut probe) | 0.655 (0.604-0.707) | 0.476 (0.423-0.527) | 0.506 (0.430-0.583) |
| Lexical overlap | 0.567 (0.528-0.606) | 0.645 (0.596-0.697) | 0.713 (0.654-0.765) |
| NLI DeBERTa-v3-small (off the shelf) | 0.564 (0.510-0.616) | 0.518 (0.488-0.552) | 0.762 (0.696-0.825) |
| Qwen2.5-1.5B base, zero-shot | 0.602 (0.550-0.654) | 0.527 (0.484-0.574) | 0.631 (0.582-0.680) |
| Qwen2.5-1.5B base, 4-shot | 0.596 (0.547-0.653) | 0.527 (0.480-0.575) | 0.738 (0.679-0.795) |
| Qwen2.5-1.5B + QLoRA SFT r=16 | 0.727 (0.681-0.773) | 0.747 (0.700-0.795) | 0.850 (0.794-0.902) |

## Transfer test: RAG question answering (never trained on)

320 rows from 140 source documents: 160 supported, 160 unsupported. MS MARCO passages from RAGTruth; held out on purpose, see docs/data.md.

| System | Balanced accuracy (95% CI) | Precision | Recall | F1 (95% CI) | AUROC (95% CI) | ECE | ms per item (device) | Size MB |
|---|---|---|---|---|---|---|---|---|
| Majority class (always unsupported) | 0.500 (0.500-0.500) | 0.500 | 1.000 | 0.667 (0.611-0.720) | 0.500 (0.500-0.500) | - | 0.0 (-) | 0 |
| Length only (shortcut probe) | 0.512 (0.503-0.528) | 0.506 | 1.000 | 0.672 (0.615-0.724) | 0.702 (0.639-0.761) | 0.181 | 0.0 (laptop CPU) | 0 |
| Lexical overlap | 0.688 (0.635-0.741) | 0.660 | 0.775 | 0.713 (0.651-0.771) | 0.765 (0.711-0.819) | - | 0.2 (laptop CPU) | 0 |
| NLI DeBERTa-v3-small (off the shelf) | 0.641 (0.602-0.680) | 0.586 | 0.963 | 0.728 (0.673-0.782) | 0.749 (0.693-0.803) | 0.425 | 159.4 (Tesla T4) | 568 |
| Qwen2.5-1.5B base, zero-shot | 0.550 (0.489-0.608) | 0.546 | 0.594 | 0.569 (0.496-0.638) | 0.570 (0.502-0.635) | 0.485 | 111.8 (Tesla T4) | 1122 |
| Qwen2.5-1.5B base, 4-shot | 0.541 (0.475-0.603) | 0.542 | 0.525 | 0.533 (0.453-0.610) | 0.568 (0.501-0.635) | 0.457 | 302.3 (Tesla T4) | 1122 |
| Qwen2.5-1.5B + QLoRA SFT r=16 | 0.634 (0.593-0.673) | 0.581 | 0.969 | 0.726 (0.676-0.776) | 0.860 (0.818-0.898) | 0.150 | 188.9 (Tesla T4) | 1196 |
| Qwen2.5-1.5B + QLoRA SFT r=8 (ablation) | not run | | | | | | | |
| Qwen2.5-1.5B + QLoRA DPO r=16 (optional) | not run | | | | | | | |
| Paid judge model | not run | | | | | | | |

## External test: human-labelled answers from a real RAG app

File `radar_human.jsonl`. 52 rows from 52 source documents: 21 supported, 31 unsupported. Small set: read the intervals, not the point values.

Overlap with train (must be 0): {'shared_groups': 0, 'shared_rows': 0, 'shared_passages': 0, 'near_duplicate_passages': 0}

| System | Balanced accuracy (95% CI) | Precision | Recall | F1 (95% CI) | AUROC (95% CI) | ECE | ms per item (device) | Size MB |
|---|---|---|---|---|---|---|---|---|
| Majority class (always unsupported) | 0.500 (0.500-0.500) | 0.596 | 1.000 | 0.747 (0.632-0.844) | 0.500 (0.500-0.500) | - | 0.0 (-) | 0 |
| Length only (shortcut probe) | 0.515 (0.443-0.599) | 0.604 | 0.935 | 0.734 (0.620-0.833) | 0.625 (0.466-0.781) | 0.092 | 0.0 (laptop CPU) | 0 |
| Lexical overlap | 0.736 (0.608-0.856) | 0.815 | 0.710 | 0.759 (0.622-0.875) | 0.768 (0.636-0.887) | - | 0.2 (laptop CPU) | 0 |
| NLI DeBERTa-v3-small (off the shelf) | 0.721 (0.592-0.840) | 0.750 | 0.871 | 0.806 (0.697-0.904) | 0.840 (0.717-0.935) | 0.299 | 159.4 (Tesla T4) | 568 |
| Qwen2.5-1.5B base, zero-shot | 0.634 (0.524-0.750) | 0.674 | 0.935 | 0.784 (0.675-0.878) | 0.836 (0.716-0.938) | 0.559 | 115.5 (Tesla T4) | 1122 |
| Qwen2.5-1.5B base, 4-shot | 0.643 (0.548-0.740) | 0.674 | 1.000 | 0.805 (0.696-0.892) | 0.879 (0.764-0.974) | 0.444 | 308.1 (Tesla T4) | 1122 |
| Qwen2.5-1.5B + QLoRA SFT r=16 | 0.643 (0.545-0.739) | 0.674 | 1.000 | 0.805 (0.698-0.894) | 0.905 (0.789-0.988) | 0.120 | 199.3 (Tesla T4) | 1196 |
| Qwen2.5-1.5B + QLoRA SFT r=8 (ablation) | not run | | | | | | | |
| Qwen2.5-1.5B + QLoRA DPO r=16 (optional) | not run | | | | | | | |
| Paid judge model | not run | | | | | | | |

## LLM verdicts as generated (main test, no tuned cut-off)

| System | Balanced accuracy | Says unsupported | Reply not parseable |
|---|---|---|---|
| Qwen2.5-1.5B base, zero-shot | 0.499 | 0.1% | 0 of 866 |
| Qwen2.5-1.5B base, 4-shot | 0.514 | 2.3% | 0 of 866 |
| Qwen2.5-1.5B + QLoRA SFT r=16 | 0.756 | 39.3% | 0 of 866 |

## Paired comparison (main test, McNemar exact test)

p values are not corrected for several comparisons, and rows that share a document are not fully independent, so treat p between 0.01 and 0.05 as weak evidence.

| A | B | only A right | only B right | p (exact) |
|---|---|---|---|---|
| qwen_sft_r16 | qwen_zero_shot | 232 | 77 | 0.0000 |
| qwen_sft_r16 | qwen_few_shot | 208 | 68 | 0.0000 |
| qwen_sft_r16 | nli_deberta_small | 256 | 105 | 0.0000 |
| qwen_sft_r16 | lexical_overlap | 225 | 109 | 0.0000 |
| qwen_sft_r16 | qwen_sft_r8 | not run | | |
| qwen_sft_r16 | qwen_dpo_r16 | not run | | |
| qwen_sft_r16 | paid_judge | not run | | |
| nli_deberta_small | lexical_overlap | 171 | 206 | 0.0798 |

## Calibration (main test)

ECE in the table above is the gap between the score and the real rate.

**Length only (shortcut probe)**

| Score bin | n | Mean score | Really unsupported |
|---|---|---|---|
| 0.2-0.3 | 3 | 0.28 | 0.67 |
| 0.3-0.4 | 78 | 0.36 | 0.42 |
| 0.4-0.5 | 358 | 0.46 | 0.44 |
| 0.5-0.6 | 337 | 0.53 | 0.50 |
| 0.6-0.7 | 69 | 0.64 | 0.80 |
| 0.7-0.8 | 21 | 0.73 | 0.86 |

**NLI DeBERTa-v3-small (off the shelf)**

| Score bin | n | Mean score | Really unsupported |
|---|---|---|---|
| 0.0-0.1 | 127 | 0.02 | 0.20 |
| 0.1-0.2 | 11 | 0.15 | 0.09 |
| 0.2-0.3 | 5 | 0.26 | 0.40 |
| 0.3-0.4 | 9 | 0.35 | 0.44 |
| 0.4-0.5 | 7 | 0.45 | 0.29 |
| 0.5-0.6 | 6 | 0.54 | 0.33 |
| 0.6-0.7 | 10 | 0.63 | 0.80 |
| 0.7-0.8 | 17 | 0.76 | 0.76 |
| 0.8-0.9 | 21 | 0.85 | 0.48 |
| 0.9-1.0 | 653 | 0.99 | 0.56 |

**Qwen2.5-1.5B base, zero-shot**

| Score bin | n | Mean score | Really unsupported |
|---|---|---|---|
| 0.0-0.1 | 832 | 0.01 | 0.49 |
| 0.1-0.2 | 23 | 0.14 | 0.83 |
| 0.2-0.3 | 8 | 0.25 | 1.00 |
| 0.3-0.4 | 2 | 0.38 | 0.50 |
| 0.6-0.7 | 1 | 0.60 | 0.00 |

**Qwen2.5-1.5B base, 4-shot**

| Score bin | n | Mean score | Really unsupported |
|---|---|---|---|
| 0.0-0.1 | 746 | 0.01 | 0.45 |
| 0.1-0.2 | 47 | 0.15 | 0.77 |
| 0.2-0.3 | 26 | 0.25 | 0.85 |
| 0.3-0.4 | 15 | 0.34 | 0.80 |
| 0.4-0.5 | 12 | 0.45 | 0.75 |
| 0.5-0.6 | 4 | 0.52 | 1.00 |
| 0.6-0.7 | 9 | 0.65 | 0.78 |
| 0.7-0.8 | 3 | 0.75 | 1.00 |
| 0.8-0.9 | 3 | 0.84 | 0.67 |
| 0.9-1.0 | 1 | 0.98 | 0.00 |

**Qwen2.5-1.5B + QLoRA SFT r=16**

| Score bin | n | Mean score | Really unsupported |
|---|---|---|---|
| 0.0-0.1 | 129 | 0.07 | 0.09 |
| 0.1-0.2 | 167 | 0.14 | 0.23 |
| 0.2-0.3 | 114 | 0.24 | 0.39 |
| 0.3-0.4 | 72 | 0.35 | 0.53 |
| 0.4-0.5 | 43 | 0.45 | 0.44 |
| 0.5-0.6 | 37 | 0.55 | 0.59 |
| 0.6-0.7 | 46 | 0.65 | 0.63 |
| 0.7-0.8 | 34 | 0.75 | 0.76 |
| 0.8-0.9 | 65 | 0.86 | 0.86 |
| 0.9-1.0 | 159 | 0.95 | 0.94 |

## Error analysis (main test)

Full texts: data/error_analysis_<system>.md (local only).

### NLI DeBERTa-v3-small (off the shelf)

- false supported (missed hallucination): 122 (hagrid 90, ragtruth_summary 12, vitaminc 20; 0 marked 'true but not in the source' by the annotators)
  - most confident: hagrid-378-0, hagrid-378-1, hagrid-12-1, hagrid-175-1, hagrid-1062-1, hagrid-339-1, hagrid-310-0, hagrid-2728-0, vitaminc-5ebe6f30c9e77c0008408abd_2, vitaminc-5ece9462c9e77c0008466b2a_2
- false unsupported (false alarm): 239 (hagrid 73, ragtruth_summary 148, vitaminc 18; 0 marked 'true but not in the source' by the annotators)
  - most confident: hagrid-1738-0, ragtruth-summary-2446, ragtruth-summary-839, ragtruth-summary-876, ragtruth-summary-837, vitaminc-5ed7ad03c9e77c00084a735b_2, hagrid-2755-0, vitaminc-5ede072ec9e77c00084d27cd_4, vitaminc-5ed4de09c9e77c000848ad5a_1, hagrid-1802-0

### Qwen2.5-1.5B base, zero-shot

- false supported (missed hallucination): 178 (hagrid 77, ragtruth_summary 100, vitaminc 1; 4 marked 'true but not in the source' by the annotators)
  - most confident: hagrid-296-1, hagrid-296-0, hagrid-2776-1, hagrid-216-1, hagrid-148-1, ragtruth-summary-605, hagrid-1714-1, ragtruth-summary-4778, hagrid-814-0, hagrid-2815-0
- false unsupported (false alarm): 187 (hagrid 72, ragtruth_summary 57, vitaminc 58; 0 marked 'true but not in the source' by the annotators)
  - most confident: hagrid-1738-0, hagrid-347-1, hagrid-1850-1, hagrid-347-0, vitaminc-5ec7d50cc9e77c00084410c0_1, hagrid-139-0, vitaminc-5ee39419c9e77c0008ccfacf_2, vitaminc-5ec7d50dc9e77c0008441c54_1, vitaminc-5ee39419c9e77c0008ccfad7_1, hagrid-2843-1

### Qwen2.5-1.5B base, 4-shot

- false supported (missed hallucination): 175 (hagrid 66, ragtruth_summary 104, vitaminc 5; 5 marked 'true but not in the source' by the annotators)
  - most confident: ragtruth-summary-4472, ragtruth-summary-208, ragtruth-summary-3866, ragtruth-summary-410, ragtruth-summary-212, hagrid-296-0, hagrid-296-1, ragtruth-summary-356, ragtruth-summary-605, hagrid-2630-0
- false unsupported (false alarm): 175 (hagrid 85, ragtruth_summary 53, vitaminc 37; 0 marked 'true but not in the source' by the annotators)
  - most confident: hagrid-1738-0, hagrid-347-1, hagrid-2843-1, hagrid-347-0, hagrid-1850-1, hagrid-2843-0, vitaminc-5ec7d50cc9e77c00084410c0_1, vitaminc-5ec7d50ec9e77c0008442572_4, hagrid-2935-0, vitaminc-5ede072ec9e77c00084d2650_1

### Qwen2.5-1.5B + QLoRA SFT r=16

- false supported (missed hallucination): 84 (hagrid 26, ragtruth_summary 56, vitaminc 2; 2 marked 'true but not in the source' by the annotators)
  - most confident: ragtruth-summary-4052, hagrid-2284-1, hagrid-2284-0, ragtruth-summary-4472, ragtruth-summary-1604, ragtruth-summary-4211, ragtruth-summary-410, ragtruth-summary-1238, ragtruth-summary-2247, ragtruth-summary-4526
- false unsupported (false alarm): 126 (hagrid 76, ragtruth_summary 28, vitaminc 22; 0 marked 'true but not in the source' by the annotators)
  - most confident: ragtruth-summary-5200, hagrid-1566-1, hagrid-1802-0, hagrid-1490-0, vitaminc-5ed4de09c9e77c000848ad5a_1, ragtruth-summary-4066, hagrid-1738-0, hagrid-1733-1, vitaminc-5ec5598bc9e77c000842c017_1, ragtruth-summary-616
