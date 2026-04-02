from typing import Dict, List, Optional

from data.schemas import PerceptionFrame, PerceptionObject


def build_perception_frame(
    objects: List[Dict[str, object]],
    image_paths: Optional[List[str]] = None,
    traffic_light_state: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
    can_bus: Optional[Dict[str, object]] = None,
) -> PerceptionFrame:
    """
    Convert raw annotations or detections into a structured PerceptionFrame.
    """
    frame_objects = [
        PerceptionObject.model_validate(obj) for obj in objects
    ]
    return PerceptionFrame(
        image_paths=image_paths or [],
        traffic_light_state=traffic_light_state,
        objects=frame_objects,
        can_bus=can_bus or {},
        metadata=metadata or {},
    )


def build_from_nuscenes_sample(
    sample: Dict[str, object],
    loader: object,
) -> PerceptionFrame:
    """
    Build a PerceptionFrame from a nuScenes sample and loader.
    The loader is expected to expose get_front_camera_paths/get_annotations.
    """
    image_paths = []
    if hasattr(loader, "get_front_camera_paths"):
        image_paths = loader.get_front_camera_paths(sample)
    objects = []
    if hasattr(loader, "get_annotations"):
        objects = loader.get_annotations(sample)
    map_context = {}
    if hasattr(loader, "get_map_context"):
        map_context = loader.get_map_context(sample)
    ego_pose = {}
    if hasattr(loader, "get_ego_pose"):
        ego_pose = loader.get_ego_pose(sample)
    can_bus = {}
    if hasattr(loader, "get_can_bus"):
        can_bus = loader.get_can_bus(sample)
    frame = build_perception_frame(
        objects=objects,
        image_paths=image_paths,
        traffic_light_state=None,
        metadata={
            "source": "nuscenes",
            "map_context": map_context,
            "ego_pose": ego_pose,
            "sample_prev": sample.get("prev"),
            "sample_next": sample.get("next"),
        },
        can_bus=can_bus,
    )
    frame.scene_id = sample.get("scene_token")
    frame.sample_token = sample.get("token")
    frame.timestamp = sample.get("timestamp")
    return frame
