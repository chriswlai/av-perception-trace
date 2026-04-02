from data.schemas import ActionType, PerceptionFrame
from teacher.rules import apply_rules


def _frame_with(overrides: dict | None = None) -> PerceptionFrame:
    payload = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "image_paths": [],
        "traffic_light_state": None,
        "objects": [],
        "uncertainty_flags": [],
        "metadata": {"map_context": {}},
    }
    if overrides:
        payload.update(overrides)
    return PerceptionFrame.model_validate(payload)


def test_rules_stop_on_red_light() -> None:
    frame = _frame_with({"traffic_light_state": "red"})
    action, confidence, constraints, targets, relations = apply_rules(frame)
    assert action == ActionType.STOP
    assert "traffic_light_red" in constraints
    assert confidence <= 0.9
    assert targets == []
    assert relations == []


def test_rules_slow_on_stop_line() -> None:
    frame = _frame_with({"metadata": {"map_context": {"stop_line_tokens": ["stop_1"]}}})
    action, _, constraints, _, _ = apply_rules(frame)
    assert action == ActionType.SLOW
    assert "stop_line_ahead" in constraints


def test_rules_slow_when_lanes_missing() -> None:
    frame = _frame_with({"metadata": {"map_context": {"lane_tokens": []}}})
    action, _, constraints, _, _ = apply_rules(frame)
    assert action == ActionType.SLOW
    assert "lane_unavailable" in constraints


def test_rules_slow_on_uncertainty() -> None:
    frame = _frame_with({"uncertainty_flags": ["low_confidence"]})
    action, _, constraints, _, _ = apply_rules(frame)
    assert action == ActionType.SLOW
    assert "perception_uncertain" in constraints


def test_rules_stop_on_close_crosswalk_pedestrian() -> None:
    frame = _frame_with(
        {
            "objects": [
                {
                    "id": "ped_1",
                    "label": "human.pedestrian.adult",
                    "distance_m": 5.0,
                    "near_crosswalk": True,
                }
            ]
        }
    )
    action, _, constraints, _, _ = apply_rules(frame)
    assert action == ActionType.STOP
    assert "pedestrian_close_near_crosswalk" in constraints


def test_rules_slow_on_closing_vehicle() -> None:
    frame = _frame_with(
        {
            "objects": [
                {
                    "id": "veh_1",
                    "label": "vehicle.car",
                    "distance_m": 15.0,
                    "approach_rate_mps": 3.0,
                }
            ]
        }
    )
    action, _, constraints, _, _ = apply_rules(frame)
    assert action == ActionType.SLOW
    assert "closing_on_vehicle_ahead" in constraints
