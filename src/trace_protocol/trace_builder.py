from typing import List

from data.schemas import ActionType, PerceptionFrame, Relation, TargetRef, TraceAction, TraceRecord
from trace_protocol.render_explanation import render_explanation


def build_trace(
    frame: PerceptionFrame,
    action: ActionType,
    confidence: float,
    constraints: List[str],
    targets: List[TargetRef],
    relations: List[Relation],
    failure_tags: List[str] | None = None,
) -> TraceRecord:
    target_labels = [_format_target(t) for t in targets]
    explanation = render_explanation(action.value, constraints, target_labels)
    return TraceRecord(
        targets=targets,
        relations=relations,
        action=TraceAction(type=action, confidence=confidence),
        constraints=constraints,
        explanation=explanation,
        can_bus=frame.can_bus,
        failure_tags=failure_tags or [],
        metadata={
            "scene_id": frame.scene_id,
            "sample_token": frame.sample_token,
            "timestamp": frame.timestamp,
        },
    )


def _format_target(target: TargetRef) -> str:
    if target.label and target.distance_m is not None:
        return f"{target.label} ({target.distance_m:.1f}m)"
    return target.label or target.id
