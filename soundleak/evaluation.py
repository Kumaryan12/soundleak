"""Post-fit evaluation: location references are never passed to the analyzer."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .physics import local_xy


def evaluate(case_directory, reference):
    path = Path(case_directory)/"report.json"
    report = json.loads(path.read_text())
    search = report.get("search")
    hypotheses = search.get("hypotheses", []) if search else []
    kind = reference.get("kind", "unknown")
    lat, lon = reference["latitude"], reference["longitude"]
    if lat is None or lon is None or not (-90 < lat < 90 and -180 <= lon <= 180):
        raise ValueError("Evaluation needs a documented finite reference latitude and longitude.")
    distances = [float(np.hypot(*local_xy(h["latitude"], h["longitude"], lat, lon))) for h in hypotheses]
    source_informed = "source-informed" in report.get("provenance", {}).get("scope", "").lower()
    # Independent evaluation is an operator assertion with explicit evidence, not
    # something a software test or a close fit can establish on its own.
    eligible = bool(kind == "surveyed_recorder" and reference.get("independent_holdout") is True
                    and reference.get("coordinate_evidence") and reference.get("recording_clock_evidence")
                    and not source_informed)
    return {"case": str(case_directory), "frozen_report_sha256": sha256(path.read_bytes()).hexdigest(),
            "reference_kind": kind, "reference_evidence": reference,
            "search_status": search.get("status") if search else "not_searched",
            "eligible_for_independent_accuracy_summary": eligible,
            "independence_basis": "Operator assertion and supplied evidence; not automatically verified.",
            "top_ranked_distance_m": distances[0] if distances else None,
            "all_retained_hypothesis_distances_m": distances,
            "diagnostic_top_ranked_within_m": {str(t): distances[0] <= t if distances else False for t in (100, 500, 1000)},
            "interpretation": "Distances to approximate public markers are diagnostics, not surveyed accuracy. An ambiguous or insufficient-evidence result is not a successful location assertion."}
