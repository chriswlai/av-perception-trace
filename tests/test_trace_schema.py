import pytest

from data.schemas import ActionType, PerceptionFrame, TraceRecord


def test_perception_frame_validation() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "image_paths": ["data/nuscenes_mini/demo/front.jpg"],
        "traffic_light_state": "green",
        "objects": [
            {
                "id": "pedestrian_1",
                "label": "pedestrian",
                "bbox": [0.45, 0.4, 0.1, 0.2],
                "confidence": 0.88,
                "distance_m": 8.0,
                "in_ego_corridor": True,
                "near_crosswalk": True,
            }
        ],
        "uncertainty_flags": [],
        "metadata": {"source": "dummy"},
    }
    parsed = PerceptionFrame.model_validate(frame)
    assert parsed.objects[0].label == "pedestrian"


def test_perception_frame_defaults() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
    }
    parsed = PerceptionFrame.model_validate(frame)
    assert parsed.image_paths == []
    assert parsed.objects == []
    assert parsed.uncertainty_flags == []
    assert parsed.can_bus.model_dump(exclude_none=True) == {}
    assert parsed.metadata == {}


def test_perception_frame_can_bus_round_trip() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "can_bus": {"vehicle_speed_mps": 1.5, "motion_state": "slowing"},
    }
    parsed = PerceptionFrame.model_validate(frame)
    assert parsed.can_bus.vehicle_speed_mps == 1.5
    assert parsed.can_bus.motion_state.value == "slowing"


def test_perception_frame_requires_object_id_and_label() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "objects": [{"label": "pedestrian"}],
    }
    with pytest.raises(ValueError):
        PerceptionFrame.model_validate(frame)


def test_perception_frame_attributes_accept_dict() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "objects": [
            {
                "id": "pedestrian_1",
                "label": "pedestrian",
                "attributes": {"visibility_token": 2},
            }
        ],
    }
    parsed = PerceptionFrame.model_validate(frame)
    assert parsed.objects[0].attributes["visibility_token"] == 2


def test_perception_frame_rejects_invalid_bbox() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "objects": [
            {
                "id": "pedestrian_1",
                "label": "pedestrian",
                "bbox": [0.1, 0.2, 0.3],
            }
        ],
    }
    with pytest.raises(ValueError):
        PerceptionFrame.model_validate(frame)


def test_perception_frame_rejects_invalid_traffic_light() -> None:
    frame = {
        "scene_id": "scene_demo",
        "sample_token": "sample_demo",
        "timestamp": 0.0,
        "traffic_light_state": "blue",
    }
    with pytest.raises(ValueError):
        PerceptionFrame.model_validate(frame)


def test_trace_schema_validation() -> None:
    record = {
        "targets": [{"id": "pedestrian_1", "label": "pedestrian"}],
        "relations": [],
        "action": {"type": ActionType.STOP, "confidence": 0.9},
        "constraints": ["pedestrian_in_path"],
        "explanation": "The vehicle stopped because pedestrian_in_path involving pedestrian_1.",
        "failure_tags": ["PERCEPTION_MISSING_OBJECT"],
        "metadata": {"scene_id": "scene_demo", "sample_token": "sample_demo", "timestamp": 0.0},
    }
    trace = TraceRecord.model_validate(record)
    assert trace.action.type == ActionType.STOP
    assert trace.action.confidence == 0.9


def test_trace_schema_rejects_invalid_constraint() -> None:
    record = {
        "targets": [{"id": "pedestrian_1", "label": "pedestrian"}],
        "relations": [],
        "action": {"type": ActionType.STOP, "confidence": 0.9},
        "constraints": ["made_up_constraint"],
        "explanation": "Invalid constraint.",
        "metadata": {"scene_id": "scene_demo", "sample_token": "sample_demo", "timestamp": 0.0},
    }
    with pytest.raises(ValueError):
        TraceRecord.model_validate(record)
