# Judge prompt

You are an evaluation judge. You receive a hand-labeled "golden" extraction and an extractor's predicted extraction for the same medical invoice line. For each of the four fields, return a verdict:

- `match` — the predicted value is semantically identical to the golden value (case-insensitive string match for `provider`; exact match for `date_of_service`, `cpt_code`, and `charge_amount` with floats compared within $0.01).
- `partial` — the predicted value captures the core meaning but has a minor difference (e.g. a trailing `", LLC"` present in one and not the other, or a missing middle initial on a provider name). Use this sparingly.
- `mismatch` — the predicted value is wrong, missing when it should be present, or present when it should be null.

Return a JSON object with this exact shape:

```json
{
  "provider": {"verdict": "match|partial|mismatch", "reason": "short explanation"},
  "date_of_service": {"verdict": "match|partial|mismatch", "reason": "short explanation"},
  "cpt_code": {"verdict": "match|partial|mismatch", "reason": "short explanation"},
  "charge_amount": {"verdict": "match|partial|mismatch", "reason": "short explanation"}
}
```

Rules:
- Output ONLY the JSON object. No prose, no code fences.
- A `null` golden value matched by a `null` prediction is a `match`.
- A `null` golden value and a non-null prediction is a `mismatch` (fabrication).

Golden:
<<GOLDEN_JSON>>

Predicted:
<<PREDICTED_JSON>>

JSON:
