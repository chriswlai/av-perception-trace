from collections import Counter
from typing import List


def render_explanation(action: str, constraints: List[str], targets: List[str]) -> str:
    action_phrase = _action_phrase(action)
    target_text = _summarize_targets(targets)
    if constraints:
        constraint_text = _summarize_constraints(constraints)
        if target_text:
            return f"The vehicle {action_phrase} because {constraint_text} involving {target_text}."
        return f"The vehicle {action_phrase} because {constraint_text}."
    return f"The vehicle {action_phrase} because no blocking constraints were detected."


def _action_phrase(action: str) -> str:
    mapping = {
        "STOP": "stopped",
        "SLOW": "slowed",
        "PROCEED": "proceeded",
    }
    return mapping.get(action, action.lower())


def _summarize_targets(targets: List[str]) -> str:
    if not targets:
        return ""
    counts = Counter(targets)
    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(label)
        else:
            parts.append(f"{label} ({count})")
    return ", ".join(parts)


def _summarize_constraints(constraints: List[str]) -> str:
    mapping = {
        "pedestrian_in_path": "a pedestrian is in the ego corridor",
        "pedestrian_near_path": "a pedestrian is near the ego corridor",
        "pedestrian_close_near_crosswalk": "a pedestrian is close at a crosswalk",
        "cyclist_in_path": "a cyclist is in the ego corridor",
        "cyclist_close_in_path": "a cyclist is close in the ego corridor",
        "vehicle_ahead_close": "a vehicle is close ahead",
        "vehicle_stopped_ahead": "a stopped vehicle is ahead",
        "closing_on_vehicle_ahead": "the ego vehicle is closing on a vehicle ahead",
        "traffic_light_red": "the traffic light is red",
        "traffic_light_yellow": "the traffic light is yellow",
        "red_light_at_stop_line": "a red light is active at the stop line",
        "yellow_light_with_vru": "a yellow light is active with vulnerable road users nearby",
        "stop_line_ahead": "a stop line is ahead",
        "lane_unavailable": "lane context is unavailable",
        "ego_path_uncertain_with_agents": "the ego path is uncertain with nearby agents",
        "perception_uncertain": "perception is uncertain",
        "uncertain_object_near_path": "an uncertain object is near the ego path",
    }
    phrases = [mapping.get(c, c.replace("_", " ")) for c in constraints]
    return ", ".join(phrases)
