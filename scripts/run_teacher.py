import argparse
import json
from pathlib import Path
from typing import List

from data.schemas import PerceptionFrame
from teacher.teacher_runner import generate_traces
from trace_protocol.trace_store import write_traces_jsonl


def _dummy_frames() -> List[PerceptionFrame]:
    return [
        PerceptionFrame.model_validate(
            {
                "scene_id": "scene_demo",
                "sample_token": "sample_demo_1",
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
        ),
        PerceptionFrame.model_validate(
            {
                "scene_id": "scene_demo",
                "sample_token": "sample_demo_2",
                "timestamp": 1.0,
                "image_paths": ["data/nuscenes_mini/demo/front.jpg"],
                "traffic_light_state": "green",
                "objects": [
                    {
                        "id": "pedestrian_2",
                        "label": "pedestrian",
                        "bbox": [0.7, 0.5, 0.05, 0.1],
                        "confidence": 0.72,
                        "distance_m": 18.0,
                        "in_ego_corridor": False,
                        "near_crosswalk": True,
                    }
                ],
                "uncertainty_flags": [],
                "metadata": {"source": "dummy"},
            }
        ),
        PerceptionFrame.model_validate(
            {
                "scene_id": "scene_demo",
                "sample_token": "sample_demo_3",
                "timestamp": 2.0,
                "image_paths": ["data/nuscenes_mini/demo/front.jpg"],
                "traffic_light_state": "green",
                "objects": [],
                "uncertainty_flags": [],
                "metadata": {"source": "dummy"},
            }
        ),
    ]


def _load_frames_jsonl(path: str | Path, limit: int) -> List[PerceptionFrame]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    frames: List[PerceptionFrame] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            frames.append(PerceptionFrame.model_validate(payload))
            if len(frames) >= limit:
                break
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/perception_frames.jsonl")
    parser.add_argument("--output", default="data/teacher_traces.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    if args.dummy:
        frames = _dummy_frames()[: args.limit]
    else:
        frames = _load_frames_jsonl(args.input, args.limit)
        if not frames:
            frames = _dummy_frames()[: args.limit]

    traces = generate_traces(frames)
    write_traces_jsonl(args.output, traces)
    print(f"Wrote {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()
