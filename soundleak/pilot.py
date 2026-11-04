"""Evaluate multiple frozen research cases with separately supplied references."""
import json
from pathlib import Path

from .archive import file_sha256
from .evaluation import evaluate


def evaluate_pilot(manifest_path, output):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("cases"), list):
        raise ValueError("Pilot manifest needs schema_version 1 and a cases list.")
    results, failures = [], []
    for entry in manifest["cases"]:
        identity = entry["id"]
        try:
            case = (manifest_path.parent/entry["case_directory"]).resolve()
            reference_path = (manifest_path.parent/entry["reference_file"]).resolve()
            frozen_hash = file_sha256(case/"report.json")
            reference = json.loads(reference_path.read_text())
            result = evaluate(case, reference)
            if result["frozen_report_sha256"] != frozen_hash or file_sha256(case/"report.json") != frozen_hash:
                raise ValueError("Case changed while evaluating the held-out reference.")
            results.append({"id": identity, **result})
        except (KeyError, ValueError, TypeError, OSError) as exc:
            failures.append({"id": identity, "status": "needs_input", "reason": str(exc)})
    eligible = [r for r in results if r["eligible_for_independent_accuracy_summary"]]
    report = {"schema_version": 1, "manifest_sha256": file_sha256(manifest_path),
              "state": "needs_input" if not results or failures else "evaluated",
              "declared_cases": len(manifest["cases"]), "evaluated_cases": len(results),
              "eligible_independent_cases": len(eligible), "cases": results, "failures": failures,
              "eligible_cases_with_returned_hypotheses": sum(r["top_ranked_distance_m"] is not None for r in eligible),
              "eligible_diagnostic_within_m": {str(t): sum(r["diagnostic_top_ranked_within_m"][str(t)] for r in eligible) for t in (100, 500, 1000)},
              "eligible_status_counts": {s: sum(r["search_status"] == s for r in eligible) for s in ("candidate_fit", "partial_fit", "ambiguous", "insufficient_evidence", "not_searched")},
              "interpretation": "References enter post-fit evaluation only. Independence is an operator assertion requiring evidence. Diagnostic distance thresholds are not a success rate; unresolved and failed cases remain in the report."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
