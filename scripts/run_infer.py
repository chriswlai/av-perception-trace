import argparse
import json
from pathlib import Path
from typing import Iterable, List

from data.nuscenes_loader import NuScenesMiniLoader
from data.schemas import PerceptionFrame, validate_perception_frame
from perception.features import build_from_nuscenes_sample
from tqdm import tqdm


def _dummy_frames() -> List[PerceptionFrame]:
    frames = []
    for idx in range(50):
        dummy_frame = {
            "scene_id": "scene_demo",
            "sample_token": f"sample_demo_{idx}",
            "timestamp": float(idx),
            "image_paths": ["data/nuscenes_mini/demo/front.jpg"],
            "traffic_light_state": "green",
            "objects": [
                {
                    "id": f"pedestrian_{idx}",
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
        frames.append(validate_perception_frame(dummy_frame))
    return frames


def _write_jsonl(path: str | Path, frames: Iterable[PerceptionFrame]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(frame.model_dump_json())
            handle.write("\n")
            count += 1
    return count


def _load_frames_from_nuscenes(dataset_root: str, limit: int) -> List[PerceptionFrame]:
    loader = NuScenesMiniLoader(dataset_root=dataset_root)
    frames: List[PerceptionFrame] = []
    for sample in loader.iter_samples():
        frames.append(build_from_nuscenes_sample(sample, loader))
        if len(frames) >= limit:
            break
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/nuscenes_mini")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="data/perception_frames.jsonl")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    if args.dummy:
        frames = _dummy_frames()[: args.limit]
    else:
        try:
            frames = _load_frames_from_nuscenes(args.dataset_root, args.limit)
        except ModuleNotFoundError:
            frames = _dummy_frames()[: args.limit]

    count = _write_jsonl(args.output, tqdm(frames, desc="Writing frames"))
    print(f"Wrote {count} PerceptionFrame records to {args.output}")


if __name__ == "__main__":
    main()
