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


def compute_action_accuracy(
    predicted: Iterable[TraceLike],
    ground_truth: Iterable[TraceLike],
) -> float:
    pred_index = _index_traces(predicted)
    truth_index = _index_traces(ground_truth)
    common = set(pred_index) & set(truth_index)
    if not common:
        return 0.0
    matches = 0
    for token in common:
        if _action_type(pred_index[token]) == _action_type(truth_index[token]):
            matches += 1
    return matches / len(common)


def compute_constraint_scores(
    predicted: Iterable[TraceLike],
    ground_truth: Iterable[TraceLike],
) -> Dict[str, float]:
    pred_index = _index_traces(predicted)
    truth_index = _index_traces(ground_truth)
    common = set(pred_index) & set(truth_index)
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for token in common:
        precision, recall, f1 = _constraint_scores(
            _constraints(pred_index[token]),
            _constraints(truth_index[token]),
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    total = len(common)
    return {
        "precision": precision_sum / total,
        "recall": recall_sum / total,
        "f1": f1_sum / total,
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


def _constraint_scores(pred: List[str], truth: List[str]) -> Tuple[float, float, float]:
    pred_set = set(pred)
    truth_set = set(truth)
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _index_traces(traces: Iterable[TraceLike]) -> Dict[str, TraceLike]:
    indexed: Dict[str, TraceLike] = {}
    for trace in traces:
        token = _sample_token(trace)
        if token:
            indexed[token] = trace
    return indexed


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


def _sample_token(trace: TraceLike) -> str:
    if isinstance(trace, TraceRecord):
        if trace.metadata:
            token = trace.metadata.get("sample_token")
            return str(token) if token else ""
        return ""
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else {}
    if isinstance(metadata, dict):
        token = metadata.get("sample_token")
        return str(token) if token else ""
    return ""
