# llm-extraction-eval

A minimal evaluation harness for an LLM-based structured-extraction task.

This is a sample project showing how to evaluate a non-deterministic LLM output against a hand-labeled golden dataset: computing precision, recall, and F1 per field, plus an overall LLM-as-judge agreement check against human labels.

## Problem

Given a free-text medical invoice line (e.g. an EOB line, a provider statement), extract structured fields:
- `provider` — the billing entity (string)
- `date_of_service` — ISO date (`YYYY-MM-DD`)
- `cpt_code` — the CPT procedure code (5-digit string) or `null`
- `charge_amount` — the billed amount (float, dollars) or `null`

## Approach

1. **Golden dataset** (`data/golden.jsonl`) — 20 hand-labeled invoice lines with the expected extraction.
2. **Extractor prompt** (`prompts/extract.md`) — the LLM prompt that produces the JSON extraction.
3. **Judge prompt** (`prompts/judge.md`) — an LLM-as-judge prompt that scores extractor output against the golden label on each field, returning a per-field verdict (`match` / `partial` / `mismatch`) and a short reason.
4. **Evaluator** (`eval.py`) — runs the extractor across the golden set, computes per-field precision / recall / F1, and compares the LLM-judge verdicts against the ground-truth matches to report judge-vs-human agreement (Cohen's kappa).
5. **Adversarial cases** (`data/adversarial.jsonl`) — 5 inputs designed to probe the weak spots: a prompt-injection string embedded in the invoice text, a line with the date in the wrong format, a line missing the CPT code, a line with two providers, and a line in Spanish.

## What this shows

- Treating LLM output as something to **evaluate** — not just something to use.
- Precision / recall thresholds set per field, because a missing CPT is worse than a sloppy provider string.
- An LLM judge that is itself checked against human labels — you don't trust the judge until you've measured it.
- A small adversarial battery, because precision on happy-path inputs is not the same as robustness.

## What's left (not in this sample)

- A real LLM client — the code stubs expect a `call_llm(prompt)` function; swap in Anthropic, OpenAI, or local.
- A fixed random seed strategy — LLM outputs vary across runs, and you'd want a per-run manifest.
- A CI hook — this is the shape of a nightly eval, not a merge-gating check.
- A richer golden set — 20 rows is a smoke test, not a benchmark. Production evals run against hundreds to thousands of examples per feature.

## How I'd extend it for a real feature

1. Build the golden set from real production data, hand-labeled by a subject-matter expert.
2. Choose acceptance thresholds with product: e.g. precision ≥ 0.95 on provider, recall ≥ 0.90 on CPT, charge_amount within ±$0.01.
3. Add a prompt-injection detection pass before extraction (or isolate the input).
4. Run the eval on every prompt change and every model change. Gate deploys on it.
5. Sample a small percentage of production outputs weekly for human re-labeling, to catch drift.

## Running

```bash
pip install -r requirements.txt
python eval.py
```

Expected output: a per-field precision/recall table, an overall F1, and the judge-vs-human agreement score.
