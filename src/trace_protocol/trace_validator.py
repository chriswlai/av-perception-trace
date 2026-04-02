from typing import Dict

from data.schemas import TraceRecord


def validate_trace(record: Dict[str, object]) -> TraceRecord:
    return TraceRecord.model_validate(record)
