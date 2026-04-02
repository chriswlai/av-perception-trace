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
