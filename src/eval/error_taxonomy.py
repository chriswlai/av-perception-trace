from enum import Enum
from typing import Dict, Iterable, Optional, Union

from data.schemas import TraceRecord

TraceLike = Union[TraceRecord, Dict[str, object]]


class ErrorType(str, Enum):
    PERCEPTION_MISSING_OBJECT = "PERCEPTION_MISSING_OBJECT"
    PERCEPTION_MISLOCALIZED = "PERCEPTION_MISLOCALIZED"
    REASONING_WRONG_ACTION = "REASONING_WRONG_ACTION"
    OVERCONFIDENT = "OVERCONFIDENT"
    UNCERTAINTY_TRIGGERED = "UNCERTAINTY_TRIGGERED"


def summarize_failure_taxonomy(
    traces: Iterable[TraceLike],
    frames_by_sample: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for trace in traces:
        tags = classify_trace(trace, frames_by_sample or {})
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def classify_trace(
    trace: TraceLike,
    frames_by_sample: Dict[str, Dict[str, object]],
) -> list[str]:
    tags: list[str] = []
    constraints = _constraints(trace)
    targets = _targets(trace)
    action = _action_type(trace)
    confidence = _action_confidence(trace)
    sample_token = _sample_token(trace)
    frame = frames_by_sample.get(sample_token, {}) if sample_token else {}

    if not targets and constraints:
        tags.append(ErrorType.PERCEPTION_MISSING_OBJECT.value)

    if targets and all(t.get("distance_m") is None for t in targets):
        tags.append(ErrorType.PERCEPTION_MISLOCALIZED.value)

    if _requires_stop(constraints) and action != "STOP":
        tags.append(ErrorType.REASONING_WRONG_ACTION.value)

    if action == "PROCEED" and _requires_slow(constraints):
        tags.append(ErrorType.REASONING_WRONG_ACTION.value)

    if action in {"STOP", "SLOW"} and confidence is not None and confidence > 0.9 and not constraints:
        tags.append(ErrorType.OVERCONFIDENT.value)

    if confidence is not None and confidence > 0.85 and _has_uncertainty(constraints, frame):
        tags.append(ErrorType.OVERCONFIDENT.value)

    if _has_uncertainty(constraints, frame):
        tags.append(ErrorType.UNCERTAINTY_TRIGGERED.value)

    return tags


def _has_uncertainty(constraints: list[str], frame: Dict[str, object]) -> bool:
    if any(c in {"perception_uncertain", "uncertain_object_near_path"} for c in constraints):
        return True
    flags = frame.get("uncertainty_flags")
    return isinstance(flags, list) and bool(flags)


def _requires_stop(constraints: list[str]) -> bool:
    stop_constraints = {
        "pedestrian_in_path",
        "cyclist_close_in_path",
        "pedestrian_close_near_crosswalk",
        "red_light_at_stop_line",
        "traffic_light_red",
    }
    return any(c in stop_constraints for c in constraints)


def _requires_slow(constraints: list[str]) -> bool:
    slow_constraints = {
        "pedestrian_near_path",
        "cyclist_in_path",
        "vehicle_ahead_close",
        "vehicle_stopped_ahead",
        "closing_on_vehicle_ahead",
        "traffic_light_yellow",
        "yellow_light_with_vru",
        "stop_line_ahead",
        "perception_uncertain",
        "uncertain_object_near_path",
        "lane_unavailable",
        "ego_path_uncertain_with_agents",
    }
    return any(c in slow_constraints for c in constraints)


def _action_type(trace: TraceLike) -> str:
    if isinstance(trace, TraceRecord):
        return trace.action.type.value
    action = trace.get("action", {}) if isinstance(trace, dict) else {}
    if isinstance(action, dict):
        action_type = action.get("type")
        return str(action_type) if action_type else ""
    return ""


def _action_confidence(trace: TraceLike) -> Optional[float]:
    if isinstance(trace, TraceRecord):
        return float(trace.action.confidence)
    action = trace.get("action", {}) if isinstance(trace, dict) else {}
    if isinstance(action, dict):
        confidence = action.get("confidence")
        if isinstance(confidence, (int, float)):
            return float(confidence)
    return None


def _constraints(trace: TraceLike) -> list[str]:
    if isinstance(trace, TraceRecord):
        return list(trace.constraints)
    value = trace.get("constraints", []) if isinstance(trace, dict) else []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _targets(trace: TraceLike) -> list[Dict[str, object]]:
    if isinstance(trace, TraceRecord):
        return [t.model_dump() for t in trace.targets]
    value = trace.get("targets", []) if isinstance(trace, dict) else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _sample_token(trace: TraceLike) -> Optional[str]:
    if isinstance(trace, TraceRecord):
        return str(trace.metadata.get("sample_token")) if trace.metadata else None
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    if isinstance(metadata, dict):
        token = metadata.get("sample_token")
        return str(token) if token else None
    return None
