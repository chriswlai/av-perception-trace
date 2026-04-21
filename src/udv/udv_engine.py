from typing import Dict, List

from data.schemas import (
    ActionType,
    DecideRecord,
    PerceptionFrame,
    UDVRecord,
    UnderstandRecord,
    VerifyCheck,
    VerifyRecord,
    validate_perception_frame,
    validate_udv_record,
)
from teacher.rules import apply_rules
from udv.udv_verify import verify_udv_record


def run_udv_reasoner(perception_frame: Dict[str, object]) -> UDVRecord:
    """
    Deterministic UDV placeholder that mirrors teacher rules.
    """
    frame = validate_perception_frame(perception_frame)
    action, confidence, constraints, targets, _ = apply_rules(frame)
    salient_objects = [t.label or t.id for t in targets]
    risks = list(constraints)
    uncertainty_summary = None
    if frame.uncertainty_flags:
        uncertainty_summary = ", ".join(frame.uncertainty_flags)
    elif constraints:
        uncertainty_summary = "none"

    understand = UnderstandRecord(
        salient_objects=salient_objects,
        risks=risks,
        uncertainty_summary=uncertainty_summary,
    )
    decide = DecideRecord(
        action=action,
        confidence=confidence,
        constraints=constraints,
    )
    verify = _build_verify_record(frame, decide)
    record = UDVRecord(understand=understand, decide=decide, verify=verify)
    return record


def validate_udv_json(payload: Dict[str, object]) -> UDVRecord:
    return validate_udv_record(payload)


def _build_verify_record(frame: PerceptionFrame, decide: DecideRecord) -> VerifyRecord:
    failures = verify_udv_record(frame, UDVRecord(understand=UnderstandRecord(), decide=decide, verify=VerifyRecord()))
    checks: List[VerifyCheck] = []
    if failures:
        checks = [VerifyCheck(description=failure, passed=False) for failure in failures]
    else:
        checks = [VerifyCheck(description="Decision consistent with constraints.", passed=True)]

    counterfactuals = _counterfactuals(decide.action, decide.constraints)
    return VerifyRecord(checks=checks, counterfactuals=counterfactuals)


def _counterfactuals(action: ActionType, constraints: List[str]) -> List[str]:
    if not constraints:
        return ["If a pedestrian entered the ego corridor, action would be STOP."]
    if action == ActionType.STOP:
        return ["If blocking constraints were absent, action would be PROCEED."]
    if action == ActionType.SLOW:
        return ["If risks cleared, action would be PROCEED."]
    return [f"If {constraints[0]} were present, action would be SLOW."]
