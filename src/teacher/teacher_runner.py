from typing import Iterable, List

from data.schemas import PerceptionFrame, TraceRecord
from eval.error_taxonomy import classify_trace
from teacher.rules import apply_rules
from trace_protocol.trace_builder import build_trace


def generate_traces(frames: Iterable[PerceptionFrame]) -> List[TraceRecord]:
    traces: List[TraceRecord] = []
    for frame in frames:
        action, confidence, constraints, targets, relations = apply_rules(frame)
        trace = build_trace(
            frame=frame,
            action=action,
            confidence=confidence,
            constraints=constraints,
            targets=targets,
            relations=relations,
        )
        sample_token = frame.sample_token
        if sample_token:
            frame_map = {str(sample_token): frame.model_dump()}
            trace.failure_tags = classify_trace(trace, frame_map)
        traces.append(trace)
    return traces
