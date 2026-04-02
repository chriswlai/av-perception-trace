from typing import Dict

from data.schemas import UDVRecord, validate_udv_record


def run_udv_reasoner(perception_frame: Dict[str, object]) -> UDVRecord:
    """
    Placeholder for LLM/UDV reasoning engine. Expects JSON output.
    """
    _ = perception_frame
    raise NotImplementedError("Connect to a model and validate UDV JSON.")


def validate_udv_json(payload: Dict[str, object]) -> UDVRecord:
    return validate_udv_record(payload)
