from typing import Dict


def in_ego_corridor(bbox: Dict[str, float]) -> bool:
    """
    Placeholder geometric check for whether a bbox is in the ego corridor.
    Replace with calibrated geometry or map-based checks later.
    """
    _ = bbox
    return False


def near_crosswalk(bbox: Dict[str, float]) -> bool:
    """
    Placeholder crosswalk proximity check.
    """
    _ = bbox
    return False
