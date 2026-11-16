"""
LLM extraction eval harness.

Runs an LLM extractor across a golden dataset of medical invoice lines,
computes per-field precision / recall / F1, and runs an LLM-as-judge to
score each prediction. Finally, measures judge-vs-human agreement
(Cohen's kappa) so we know whether the judge can be trusted.

Usage:
    # with a real LLM (requires ANTHROPIC_API_KEY)
    python eval.py

    # with a mock extractor (deterministic, no API key needed)
    python eval.py --mock
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent
GOLDEN_PATH = ROOT / "data" / "golden.jsonl"
ADVERSARIAL_PATH = ROOT / "data" / "adversarial.jsonl"
EXTRACT_PROMPT = (ROOT / "prompts" / "extract.md").read_text()
JUDGE_PROMPT = (ROOT / "prompts" / "judge.md").read_text()

FIELDS = ["provider", "date_of_service", "cpt_code", "charge_amount"]


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def call_llm_anthropic(prompt: str) -> str:
    """Thin wrapper over the Anthropic client. Swap for your provider of choice."""
    from anthropic import Anthropic  # local import so --mock runs without the dep installed

    client = Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def call_llm_mock(prompt: str) -> str:
    """A deterministic regex-based extractor used when no API key is present.

    This exists only so the harness runs end-to-end without external
    dependencies. It intentionally has gaps so the eval surfaces real
    precision/recall failures to look at.
    """
    if "You are an evaluation judge" in prompt:
        # For the judge, return a plausible all-match verdict most of the time.
        # This is intentionally naive so the judge-vs-human agreement check
        # produces an interesting (non-perfect) kappa.
        return json.dumps({
            f: {"verdict": "match", "reason": "mock judge: trusting the extractor"}
            for f in FIELDS
        })

    # Extractor path: pull the input out of the prompt template.
    m = re.search(r"Invoice line:\n(.+?)\nJSON:", prompt, re.DOTALL)
    text = m.group(1) if m else ""

    # Provider: everything before the first delimiter or date
    provider = None
    date_match = re.search(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text)
    if date_match:
        provider = re.split(r"[-|/\\]{2,}|  +|\s-\s", text[: date_match.start()])[0].strip(" :;|")

    # Date
    date_of_service = None
    if date_match:
        d = date_match.group(1).replace("/", "-")
        parts = d.split("-")
        if len(parts[0]) == 4:
            date_of_service = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        else:
            yy = parts[2]
            yyyy = yy if len(yy) == 4 else f"20{yy}"
            date_of_service = f"{yyyy}-{int(parts[0]):02d}-{int(parts[1]):02d}"

    # CPT: 5-digit code preceded by "CPT" or standalone
    cpt_code = None
    cpt_match = re.search(r"\b(\d{5})\b", text)
    if cpt_match and "20" not in cpt_match.group(1)[:2]:  # weak filter against years
        cpt_code = cpt_match.group(1)

    # Charge: a $ amount or a plain number after "USD"
    charge_amount = None
    amt_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text) or re.search(r"USD\s*([\d,]+(?:\.\d+)?)", text)
    if amt_match:
        charge_amount = float(amt_match.group(1).replace(",", ""))

    return json.dumps({
        "provider": provider,
        "date_of_service": date_of_service,
        "cpt_code": cpt_code,
        "charge_amount": charge_amount,
    })


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class FieldStats:
    tp: int = 0  # correct non-null prediction
    fp: int = 0  # wrong or fabricated prediction
    fn: int = 0  # missed (null when should have been non-null)
    tn: int = 0  # correctly predicted null

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def values_match(field_name: str, predicted: Any, golden: Any) -> bool:
    if predicted is None and golden is None:
        return True
    if predicted is None or golden is None:
        return False
    if field_name == "provider":
        return _norm_provider(str(predicted)) == _norm_provider(str(golden))
    if field_name == "charge_amount":
        try:
            return abs(float(predicted) - float(golden)) < 0.01
        except (TypeError, ValueError):
            return False
    return str(predicted).strip() == str(golden).strip()


def _norm_provider(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def score_prediction(predicted: dict, golden: dict, stats: dict[str, FieldStats]) -> dict[str, bool]:
    """Update stats in place, return per-field correctness for the judge comparison."""
    correctness = {}
    for f in FIELDS:
        p_val, g_val = predicted.get(f), golden.get(f)
        if g_val is None and p_val is None:
            stats[f].tn += 1
            correctness[f] = True
        elif g_val is None and p_val is not None:
            stats[f].fp += 1  # fabrication
            correctness[f] = False
        elif g_val is not None and p_val is None:
            stats[f].fn += 1
            correctness[f] = False
        elif values_match(f, p_val, g_val):
            stats[f].tp += 1
            correctness[f] = True
        else:
            stats[f].fp += 1
            correctness[f] = False
    return correctness


# ---------------------------------------------------------------------------
# Judge agreement
# ---------------------------------------------------------------------------

def cohens_kappa(human: list[bool], judge: list[bool]) -> float:
    """Cohen's kappa between two binary labelers."""
    assert len(human) == len(judge)
    n = len(human)
    if n == 0:
        return 0.0
    agree = sum(1 for h, j in zip(human, judge) if h == j)
    p_o = agree / n
    p_h = sum(human) / n
    p_j = sum(judge) / n
    p_e = p_h * p_j + (1 - p_h) * (1 - p_j)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval(call_llm, dataset: list[dict], label: str) -> None:
    print(f"\n{'=' * 60}\n  {label}  ({len(dataset)} examples)\n{'=' * 60}")
    stats = {f: FieldStats() for f in FIELDS}
    human_labels: list[bool] = []
    judge_labels: list[bool] = []
    failures: list[tuple[str, dict, dict]] = []

    for ex in dataset:
        extract_prompt = EXTRACT_PROMPT.replace("<<INPUT>>", ex["input"])
        try:
            predicted = json.loads(call_llm(extract_prompt))
        except json.JSONDecodeError:
            # A production harness would log + retry with repair. For a sample, mark all fields wrong.
            predicted = {f: None for f in FIELDS}

        correctness = score_prediction(predicted, ex["expected"], stats)

        # Judge run: compare predicted to golden.
        judge_prompt = (
            JUDGE_PROMPT
            .replace("<<GOLDEN_JSON>>", json.dumps(ex["expected"], indent=2))
            .replace("<<PREDICTED_JSON>>", json.dumps(predicted, indent=2))
        )
        try:
            verdicts = json.loads(call_llm(judge_prompt))
            for f in FIELDS:
                judge_says_match = verdicts.get(f, {}).get("verdict") == "match"
                human_labels.append(correctness[f])
                judge_labels.append(judge_says_match)
        except json.JSONDecodeError:
            # Don't poison the kappa with malformed judge output.
            pass

        if not all(correctness.values()):
            failures.append((ex["id"], ex["expected"], predicted))

    # Per-field table
    print(f"{'field':<20} {'precision':>10} {'recall':>8} {'f1':>8}")
    print("-" * 48)
    for f in FIELDS:
        s = stats[f]
        print(f"{f:<20} {s.precision():>10.3f} {s.recall():>8.3f} {s.f1():>8.3f}")

    # Overall micro-F1 (precision/recall pooled across fields)
    tp = sum(s.tp for s in stats.values())
    fp = sum(s.fp for s in stats.values())
    fn = sum(s.fn for s in stats.values())
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print("-" * 48)
    print(f"{'overall (micro)':<20} {p:>10.3f} {r:>8.3f} {f1:>8.3f}")

    # Judge agreement
    if human_labels:
        kappa = cohens_kappa(human_labels, judge_labels)
        agree_rate = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(human_labels)
        print(f"\nJudge vs. human: {agree_rate:.1%} raw agreement, kappa = {kappa:.3f}")
        if kappa < 0.6:
            print("  ⚠ kappa < 0.6 — don't trust this judge yet; recalibrate or hand-label more.")

    # Failure examples (capped)
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for fid, expected, predicted in failures[:5]:
            print(f"  {fid}")
            for f in FIELDS:
                if not values_match(f, predicted.get(f), expected.get(f)):
                    print(f"    {f}: expected={expected.get(f)!r}  got={predicted.get(f)!r}")
        if len(failures) > 5:
            print(f"  ... ({len(failures) - 5} more)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="use the deterministic mock extractor (no API key needed)")
    args = parser.parse_args()

    if args.mock or not os.getenv("ANTHROPIC_API_KEY"):
        if not args.mock:
            print("ANTHROPIC_API_KEY not set — falling back to --mock. Run with the key set for real LLM metrics.\n")
        call_llm = call_llm_mock
    else:
        call_llm = call_llm_anthropic

    run_eval(call_llm, load_jsonl(GOLDEN_PATH), "GOLDEN SET")
    run_eval(call_llm, load_jsonl(ADVERSARIAL_PATH), "ADVERSARIAL SET")
    return 0


if __name__ == "__main__":
    sys.exit(main())
