from typing import Dict, List, Tuple

from data.schemas import ActionType, PerceptionFrame, Relation, TargetRef


def apply_rules(
    frame: PerceptionFrame,
) -> Tuple[ActionType, float, List[str], List[TargetRef], List[Relation]]:
    targets_by_id: Dict[str, TargetRef] = {}
    relations: List[Relation] = []
    constraints: List[str] = []
    map_context = _map_context(frame)
    traffic_state = _traffic_light_state(frame)
    stop_lines = map_context.get("stop_line_tokens", [])
    lanes = map_context.get("lane_tokens", [])

    action = ActionType.PROCEED
    confidence = 0.8

    def propose(next_action: ActionType, next_confidence: float) -> None:
        nonlocal action, confidence
        if _action_rank(next_action) > _action_rank(action):
            action = next_action
            confidence = next_confidence
        elif next_action == action:
            confidence = max(confidence, next_confidence)

    def add_target(obj: object) -> None:
        target = _target_from_obj(obj)
        if target.id not in targets_by_id:
            targets_by_id[target.id] = target

    pedestrians_in_path = [
        obj
        for obj in frame.objects
        if _is_pedestrian(obj) and obj.in_ego_corridor
    ]
    pedestrians_near = [
        obj
        for obj in frame.objects
        if _is_pedestrian(obj) and obj.near_crosswalk
    ]
    pedestrians_near_close = [
        obj
        for obj in pedestrians_near
        if _is_close(obj, 6.0)
    ]
    cyclists_in_path = [
        obj
        for obj in frame.objects
        if _is_cyclist(obj) and obj.in_ego_corridor
    ]
    cyclists_close_in_path = [
        obj
        for obj in cyclists_in_path
        if _is_close(obj, 8.0)
    ]
    vehicles_ahead_close = [
        obj
        for obj in frame.objects
        if _is_vehicle(obj) and _is_close(obj, 12.0)
    ]
    stopped_vehicles_ahead = [
        obj
        for obj in frame.objects
        if _is_vehicle(obj)
        and _is_close(obj, 10.0)
        and obj.velocity_mps is not None
        and abs(obj.velocity_mps) <= 0.5
    ]
    closing_vehicles_ahead = [
        obj
        for obj in frame.objects
        if _is_vehicle(obj)
        and _is_close(obj, 18.0)
        and obj.approach_rate_mps is not None
        and obj.approach_rate_mps >= 2.0
    ]
    uncertain_near_path = [
        obj
        for obj in frame.objects
        if obj.in_ego_corridor
        and _is_close(obj, 15.0)
        and (
            (obj.attributes and "uncertain" in obj.attributes)
            or (obj.confidence is not None and obj.confidence < 0.45)
        )
    ]
    nearby_agents = [
        obj
        for obj in frame.objects
        if _is_close(obj, 15.0)
        and (_is_pedestrian(obj) or _is_cyclist(obj) or _is_vehicle(obj))
    ]

    if pedestrians_in_path:
        primary = pedestrians_in_path[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="in_ego_corridor", object_id="ego"))
        constraints.append("pedestrian_in_path")
        propose(ActionType.STOP, 0.9)

    if cyclists_close_in_path:
        primary = cyclists_close_in_path[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="in_ego_corridor", object_id="ego"))
        constraints.append("cyclist_close_in_path")
        propose(ActionType.STOP, 0.88)

    if traffic_state == "red" and stop_lines:
        constraints.append("red_light_at_stop_line")
        propose(ActionType.STOP, 0.95)

    if pedestrians_near_close:
        primary = pedestrians_near_close[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="near_crosswalk", object_id="ego"))
        constraints.append("pedestrian_close_near_crosswalk")
        propose(ActionType.STOP, 0.85)

    if traffic_state == "yellow" and (pedestrians_near or cyclists_in_path):
        if pedestrians_near:
            add_target(pedestrians_near[0])
            relations.append(
                Relation(subject_id=pedestrians_near[0].id, relation="near_crosswalk", object_id="ego")
            )
        if cyclists_in_path:
            add_target(cyclists_in_path[0])
            relations.append(
                Relation(subject_id=cyclists_in_path[0].id, relation="in_ego_corridor", object_id="ego")
            )
        constraints.append("yellow_light_with_vru")
        propose(ActionType.SLOW, 0.7)

    if stopped_vehicles_ahead:
        primary = stopped_vehicles_ahead[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="ahead_of", object_id="ego"))
        constraints.append("vehicle_stopped_ahead")
        propose(ActionType.SLOW, 0.6)

    if closing_vehicles_ahead:
        primary = closing_vehicles_ahead[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="ahead_of", object_id="ego"))
        constraints.append("closing_on_vehicle_ahead")
        propose(ActionType.SLOW, 0.6)

    if vehicles_ahead_close:
        primary = vehicles_ahead_close[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="ahead_of", object_id="ego"))
        constraints.append("vehicle_ahead_close")
        propose(ActionType.SLOW, 0.5)

    if pedestrians_near:
        primary = pedestrians_near[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="near_crosswalk", object_id="ego"))
        constraints.append("pedestrian_near_path")
        propose(ActionType.SLOW, 0.6)

    if cyclists_in_path:
        primary = cyclists_in_path[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="in_ego_corridor", object_id="ego"))
        constraints.append("cyclist_in_path")
        propose(ActionType.SLOW, 0.6)

    if stop_lines:
        constraints.append("stop_line_ahead")
        propose(ActionType.SLOW, 0.55)

    if traffic_state == "red":
        constraints.append("traffic_light_red")
        propose(ActionType.STOP, 0.8)

    if traffic_state == "yellow":
        constraints.append("traffic_light_yellow")
        propose(ActionType.SLOW, 0.6)

    if uncertain_near_path:
        primary = uncertain_near_path[0]
        add_target(primary)
        relations.append(Relation(subject_id=primary.id, relation="in_ego_corridor", object_id="ego"))
        constraints.append("uncertain_object_near_path")
        propose(ActionType.SLOW, 0.55)

    if frame.uncertainty_flags:
        constraints.append("perception_uncertain")
        propose(ActionType.SLOW, 0.4)

    if not lanes and nearby_agents:
        constraints.append("ego_path_uncertain_with_agents")
        propose(ActionType.SLOW, 0.45)

    if not lanes:
        constraints.append("lane_unavailable")
        propose(ActionType.SLOW, 0.4)

    return action, confidence, constraints, list(targets_by_id.values()), relations


def _map_context(frame: PerceptionFrame) -> dict:
    metadata = frame.metadata or {}
    if isinstance(metadata, dict):
        context = metadata.get("map_context")
        if isinstance(context, dict):
            return context
    return {}


def _traffic_light_state(frame: PerceptionFrame) -> str:
    state = frame.traffic_light_state
    if state is None:
        return ""
    if hasattr(state, "value"):
        return str(state.value).lower()
    if isinstance(state, str):
        return state.lower()
    return ""


def _is_pedestrian(obj: object) -> bool:
    return hasattr(obj, "label") and str(obj.label).startswith("human.pedestrian")


def _is_cyclist(obj: object) -> bool:
    return hasattr(obj, "label") and str(obj.label) in {"vehicle.bicycle", "vehicle.motorcycle"}


def _is_vehicle(obj: object) -> bool:
    return hasattr(obj, "label") and str(obj.label).startswith("vehicle.")


def _is_close(obj: object, threshold: float) -> bool:
    if not hasattr(obj, "distance_m"):
        return False
    distance = obj.distance_m
    return isinstance(distance, (int, float)) and distance <= threshold


def _action_rank(action: ActionType) -> int:
    mapping = {
        ActionType.PROCEED: 0,
        ActionType.SLOW: 1,
        ActionType.STOP: 2,
    }
    return mapping[action]


def _target_from_obj(obj: object) -> TargetRef:
    return TargetRef(
        id=obj.id,
        label=obj.label,
        distance_m=obj.distance_m,
        instance_token=obj.instance_token,
        category_name=obj.category_name,
        velocity_mps=obj.velocity_mps,
        approach_rate_mps=obj.approach_rate_mps,
        time_delta_s=obj.time_delta_s,
        yaw=obj.yaw,
        speed_class=obj.speed_class,
        size=obj.size,
        rotation=obj.rotation,
        attributes=obj.attributes,
    )
