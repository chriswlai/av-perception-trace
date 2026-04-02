from typing import Dict, Iterable, List, Tuple, Union

from data.schemas import TraceRecord

TraceLike = Union[TraceRecord, Dict[str, object]]


def summarize_action_distribution(traces: Iterable[TraceLike]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for trace in traces:
        action = _action_type(trace)
        if not action:
            continue
        counts[action] = counts.get(action, 0) + 1
    return counts


def summarize_constraint_distribution(traces: Iterable[TraceLike]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for trace in traces:
        for constraint in _constraints(trace):
            counts[constraint] = counts.get(constraint, 0) + 1
    return counts


def summarize_trace_coverage(traces: Iterable[TraceLike]) -> Dict[str, int]:
    totals = {"with_constraints": 0, "with_targets": 0, "with_relations": 0}
    for trace in traces:
        if _constraints(trace):
            totals["with_constraints"] += 1
        if _targets(trace):
            totals["with_targets"] += 1
        if _relations(trace):
            totals["with_relations"] += 1
    return totals


def summarize_confidence_stats(traces: Iterable[TraceLike]) -> Dict[str, float]:
    values: List[float] = []
    for trace in traces:
        confidence = _action_confidence(trace)
        if confidence is not None:
            values.append(confidence)
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _action_type(trace: TraceLike) -> str:
    if isinstance(trace, TraceRecord):
        return trace.action.type.value
    action = trace.get("action", {}) if isinstance(trace, dict) else {}
    if isinstance(action, dict):
        action_type = action.get("type")
        return str(action_type) if action_type else ""
    return ""


def _action_confidence(trace: TraceLike) -> float | None:
    if isinstance(trace, TraceRecord):
        return float(trace.action.confidence)
    action = trace.get("action", {}) if isinstance(trace, dict) else {}
    if isinstance(action, dict):
        confidence = action.get("confidence")
        if isinstance(confidence, (int, float)):
            return float(confidence)
    return None


def _constraints(trace: TraceLike) -> List[str]:
    if isinstance(trace, TraceRecord):
        return list(trace.constraints)
    value = trace.get("constraints", []) if isinstance(trace, dict) else []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _targets(trace: TraceLike) -> List[Dict[str, object]]:
    if isinstance(trace, TraceRecord):
        return [t.model_dump() for t in trace.targets]
    value = trace.get("targets", []) if isinstance(trace, dict) else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _relations(trace: TraceLike) -> List[Dict[str, object]]:
    if isinstance(trace, TraceRecord):
        return [r.model_dump() for r in trace.relations]
    value = trace.get("relations", []) if isinstance(trace, dict) else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []
