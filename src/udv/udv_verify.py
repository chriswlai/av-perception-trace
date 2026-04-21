from typing import List

from data.schemas import PerceptionFrame, UDVRecord


def verify_udv_record(frame: PerceptionFrame, record: UDVRecord) -> List[str]:
    """
    Lightweight validation rules for UDV outputs.
    Returns a list of verification failure strings.
    """
    failures: List[str] = []
    if record.decide.action in {"STOP", "SLOW"} and not record.decide.constraints:
        failures.append("STOP/SLOW requires at least one constraint.")
    if frame.uncertainty_flags and record.decide.confidence > 0.8:
        failures.append("High confidence despite uncertainty flags.")
    return failures


def score_udv_record(frame: PerceptionFrame, record: UDVRecord) -> float:
    failures = verify_udv_record(frame, record)
    if not record.verify.checks:
        return 1.0 if not failures else 0.0
    passed = sum(1 for check in record.verify.checks if check.passed is True)
    total = len(record.verify.checks)
    base_score = passed / total if total else 1.0
    penalty = 0.1 * len(failures)
    return max(0.0, base_score - penalty)
