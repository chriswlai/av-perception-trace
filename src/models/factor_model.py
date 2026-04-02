from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FactorModelOutput:
    action: str
    action_confidence: float
    constraints: List[str]


class FactorModel:
    """
    Placeholder model for predicting action + constraint factors.
    """

    def predict(self, features: Dict[str, float]) -> FactorModelOutput:
        _ = features
        return FactorModelOutput(action="PROCEED", action_confidence=0.5, constraints=[])


def extract_features(frame: Dict[str, object]) -> Dict[str, float]:
    objects = frame.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    metadata = frame.get("metadata", {})
    map_context = metadata.get("map_context", {}) if isinstance(metadata, dict) else {}
    can_bus = frame.get("can_bus", {})
    if not isinstance(can_bus, dict):
        can_bus = {}

    def is_ped(obj: Dict[str, object]) -> bool:
        return str(obj.get("label", "")).startswith("human.pedestrian")

    def is_cyclist(obj: Dict[str, object]) -> bool:
        return str(obj.get("label", "")) in {"vehicle.bicycle", "vehicle.motorcycle"}

    def is_vehicle(obj: Dict[str, object]) -> bool:
        return str(obj.get("label", "")).startswith("vehicle.")

    def is_close(obj: Dict[str, object], threshold: float) -> bool:
        distance = obj.get("distance_m")
        return isinstance(distance, (int, float)) and distance <= threshold

    pedestrians_in_path = [
        obj for obj in objects if is_ped(obj) and obj.get("in_ego_corridor")
    ]
    pedestrians_near = [
        obj for obj in objects if is_ped(obj) and obj.get("near_crosswalk")
    ]
    cyclists_in_path = [
        obj for obj in objects if is_cyclist(obj) and obj.get("in_ego_corridor")
    ]
    vehicles_close = [obj for obj in objects if is_vehicle(obj) and is_close(obj, 12.0)]
    closing_vehicles = [
        obj
        for obj in objects
        if is_vehicle(obj)
        and is_close(obj, 18.0)
        and isinstance(obj.get("approach_rate_mps"), (int, float))
        and obj.get("approach_rate_mps", 0.0) >= 2.0
    ]

    traffic_state = str(frame.get("traffic_light_state", "")).lower()
    uncertainty_flags = frame.get("uncertainty_flags", [])
    if not isinstance(uncertainty_flags, list):
        uncertainty_flags = []

    lane_tokens = map_context.get("lane_tokens", []) if isinstance(map_context, dict) else []
    stop_lines = map_context.get("stop_line_tokens", []) if isinstance(map_context, dict) else []
    crosswalks = (
        map_context.get("crosswalk_tokens", []) if isinstance(map_context, dict) else []
    )

    speed = can_bus.get("vehicle_speed_mps")
    if speed is None:
        speed = can_bus.get("vehicle_speed")

    return {
        "ped_in_path": float(len(pedestrians_in_path)),
        "ped_near": float(len(pedestrians_near)),
        "cyclist_in_path": float(len(cyclists_in_path)),
        "vehicle_close": float(len(vehicles_close)),
        "vehicle_closing": float(len(closing_vehicles)),
        "traffic_red": 1.0 if traffic_state == "red" else 0.0,
        "traffic_yellow": 1.0 if traffic_state == "yellow" else 0.0,
        "lanes_present": 1.0 if lane_tokens else 0.0,
        "stop_lines": float(len(stop_lines)),
        "crosswalks": float(len(crosswalks)),
        "uncertainty_flags": float(len(uncertainty_flags)),
        "can_bus_speed": float(speed) if isinstance(speed, (int, float)) else 0.0,
    }
