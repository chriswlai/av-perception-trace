from data.schemas import ActionType, UDVRecord


def test_udv_schema_validation() -> None:
    record = {
        "understand": {
            "salient_objects": ["pedestrian_1"],
            "risks": ["pedestrian_in_path"],
            "uncertainty_summary": "low",
        },
        "decide": {
            "action": ActionType.STOP,
            "confidence": 0.85,
            "constraints": ["pedestrian_in_path"],
        },
        "verify": {
            "checks": [{"description": "Stop if pedestrian in path", "passed": True}],
            "counterfactuals": ["If pedestrian were absent, action would be PROCEED."],
        },
    }
    udv = UDVRecord.model_validate(record)
    assert udv.decide.action == ActionType.STOP
