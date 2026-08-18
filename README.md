# Llama 3.2 QLoRA + RAG — Validation Results

This repository contains the validation results for the **QLoRA-fine-tuned Llama 3.2 3B Instruct** model with **RAG-augmented inference** on the hybrid bug-repair dataset.

## What was done

1. Fine-tuned **Llama 3.2 3B Instruct** on the hybrid bug-repair dataset using **QLoRA** (4-bit NF4 quantization, LoRA r=16, alpha=16, dropout=0.0).
2. Built a **FAISS retrieval index** from the training split using **all-MiniLM-L6-v2** embeddings of buggy code.
3. For each validation sample, retrieved the **top-1 most similar buggy/fixed pair** from the training set.
4. Injected the retrieved example into the prompt as reference and generated a repair patch.
5. Computed exact match, ROUGE, SacreBLEU, and similarity metrics.

## Environment

Key Python packages used:

- `transformers`
- `peft`
- `bitsandbytes`
- `sentence-transformers`
- `faiss-cpu`
- `rouge-score`
- `sacrebleu`
- `Levenshtein`
- `torch`

No fixed version pins are provided. Results may vary slightly with different package versions.

## Dataset

The hybrid bug-repair dataset combines multiple Python bug-fix sources. This validation used the `val_resplit.csv` split:

- Validation samples after filtering: **2,510**
- Filtering: dropped rows with missing `buggy_code` or `fixed_code`, and rows where `buggy_code == fixed_code`
- Each row includes a `bug_type` label used for per-category evaluation

Training split was used only for retrieval, with **20,077** rows after the same filtering.

## Configuration

| Component | Setting |
|---|---|
| Base model | `Llama 3.2 3B Instruct` |
| Fine-tuning | QLoRA, 4-bit NF4, double quantization, fp16 compute |
| LoRA | r=16, alpha=16, dropout=0.0, target modules: q,k,v,o,gate,up,down |
| Retrieval encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval index | FAISS IndexFlatIP, cosine similarity |
| Retrieval corpus | Train split, buggy-code-only embeddings |
| Top-k retrieved | 1 |
| Validation samples | 2,510 |

## Prompt used for RAG inference

System message:

```
Fix the target buggy code. References are unrelated examples. Keep the target signature and style exactly as given.
```

User message template:

```
Reference buggy 1:
<retrieved_buggy_code>
Reference fixed 1:
<retrieved_fixed_code>

Target buggy code:
```python
<target_buggy_code>
```

Return only the corrected target code.


## Overall Results

| Metric | Value |
|---|---|
| Exact Match | 21.87% |
| SacreBLEU | 85.70 |
| ROUGE-1 | 0.8731 |
| ROUGE-2 | 0.8098 |
| ROUGE-L | 0.8690 |
| Similarity ratio | 0.8861 |

## Bug Type Performance

| Bug type | n | Exact Match | ROUGE-1 | ROUGE-2 | ROUGE-L | SacreBLEU |
|---|---|---|---|---|---|---|
| other | 1547 | 2.00% | 0.8376 | 0.7521 | 0.8319 | 75.84 |
| unknown | 480 | 70.63% | 0.9730 | 0.9711 | 0.9726 | 88.39 |
| algorithm | 107 | 57.94% | 0.9182 | 0.8963 | 0.9167 | 80.08 |
| logic | 87 | 8.05% | 0.8321 | 0.7450 | 0.8226 | 82.34 |
| type | 70 | 1.43% | 0.8375 | 0.7107 | 0.8368 | 76.81 |
| build_package_merge | 56 | 82.14% | 0.9901 | 0.9900 | 0.9901 | 96.06 |
| assignment | 56 | 35.71% | 0.9072 | 0.8577 | 0.9047 | 92.23 |
| checking | 54 | 48.15% | 0.9152 | 0.8905 | 0.9125 | 76.53 |
| timing_serialization | 27 | 40.74% | 0.8628 | 0.8175 | 0.8628 | 82.05 |
| reference | 16 | 37.50% | 0.8872 | 0.8461 | 0.8872 | 73.11 |
| syntax | 10 | 0.00% | 0.6244 | 0.5269 | 0.6191 | 46.61 |

## Known limitations

- RAG was applied **at inference only**; the model was not fine-tuned with retrieved examples in training.
- Top-1 retrieval only; no top-k comparison.
- Retrieval encoder is a general sentence transformer, not fine-tuned on code.
- Performance is very poor on `syntax` and `type` bug categories.
- The `other` category dominates the validation set and has near-zero exact match, pulling down the overall result.
- Validation loss increased in later training steps, indicating overfitting in the fine-tuned model.

## How to reproduce

1. Load the QLoRA-fine-tuned Llama 3.2 3B adapter.
2. Build a FAISS index from the training split using `all-MiniLM-L6-v2` embeddings of buggy code.
3. For each validation sample, retrieve the top-1 most similar training buggy/fixed pair.
4. Build the RAG prompt using the template above.
5. Generate with `max_new_tokens=256`, `do_sample=False`.
6. Compute metrics using `rouge-score`, `sacrebleu`, and `Levenshtein`.

## Files

- `rag_val_results.csv` — reference and predicted patches for all 2,510 validation samples.
- `rag_val_metrics.csv` — overall metrics.
- `rag_val_bug_type_metrics.csv` — metrics grouped by bug type.
- `loss_progression.csv` — training and validation loss per step.
- `loss_progression.png` — loss curve plot.

## Context

Conducted as part of the CaliDebug thesis work on confidence-guided debugging.
