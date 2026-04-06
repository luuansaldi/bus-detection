"""
Candidate filter and scorer.

Takes all raw OCR candidates across zones and variants, scores them,
filters overlays, and returns the single best fleet number or None.

Scoring rules:
  base score  = EasyOCR confidence
  +0.20       if exactly 3 digits
  +0.10       if exactly 4 digits
  -0.30       if only 2 digits
  discard     if outside FLEET_MIN–FLEET_MAX
  discard     if 4 digits and >= FLEET_YEAR_THRESHOLD (e.g. 2026 from timestamp)
"""

from dataclasses import dataclass

from ocr.reader import RawCandidate
from config.settings import FLEET_MIN, FLEET_MAX, FLEET_YEAR_THRESHOLD, OCR_MIN_CONFIDENCE


@dataclass
class ScoredCandidate:
    number: int
    score: float
    zone_name: str
    variant_name: str
    raw_confidence: float


def score_and_filter(candidates: list[RawCandidate]) -> list[ScoredCandidate]:
    """Score all candidates and return only valid ones, sorted best-first."""
    scored: list[ScoredCandidate] = []

    for c in candidates:
        try:
            value = int(c.text)
        except ValueError:
            continue

        digits = len(c.text)

        # Hard reject: low OCR confidence — likely background noise or partial read
        if c.confidence < OCR_MIN_CONFIDENCE:
            continue

        # Hard reject: years from CCTV timestamp overlay (e.g. "2026")
        if digits == 4 and value >= FLEET_YEAR_THRESHOLD:
            continue

        # Hard reject: outside valid fleet range
        if not (FLEET_MIN <= value <= FLEET_MAX):
            continue

        score = c.confidence
        if digits == 3:
            score += 0.20
        elif digits == 4:
            score += 0.20  # same bonus as 3-digit — partial reads (e.g. '102' from '1022') no longer outrank
        elif digits == 2:
            score -= 0.30

        scored.append(
            ScoredCandidate(
                number=value,
                score=score,
                zone_name=c.zone_name,
                variant_name=c.variant_name,
                raw_confidence=c.confidence,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)

    # If a shorter number is a prefix of a longer one (e.g. '102' vs '1022'),
    # remove the shorter — it's almost certainly a partial OCR read of the longer.
    numbers = [c.number for c in scored]
    filtered = []
    for c in scored:
        is_partial = any(
            str(other) != str(c.number) and str(other).startswith(str(c.number))
            for other in numbers
        )
        if not is_partial:
            filtered.append(c)

    return filtered if filtered else scored


def select_best(candidates: list[RawCandidate]) -> ScoredCandidate | None:
    """
    Return the single best fleet number candidate, or None if nothing valid.
    Never returns a fallback value — silence is better than a wrong number.
    """
    valid = score_and_filter(candidates)
    return valid[0] if valid else None
