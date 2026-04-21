import argparse
import json
from pathlib import Path
from typing import Dict, List

from data.schemas import validate_perception_frame
from udv.udv_engine import run_udv_reasoner
from udv.udv_verify import score_udv_record


def _load_jsonl(path: str | Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    input_path = Path(path)
    if not input_path.exists():
        return records
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: str | Path, records: List[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/perception_frames.jsonl")
    parser.add_argument("--output", default="data/udv_outputs.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    frames = _load_jsonl(args.frames)[: args.limit]
    if not frames:
        raise RuntimeError("Frames are empty. Generate them first.")

    outputs: List[Dict[str, object]] = []
    for frame in frames:
        udv_record = run_udv_reasoner(frame)
        frame_model = validate_perception_frame(frame)
        payload = udv_record.model_dump()
        payload["verify_score"] = score_udv_record(frame_model, udv_record)
        payload["metadata"] = {
            "scene_id": frame.get("scene_id"),
            "sample_token": frame.get("sample_token"),
            "timestamp": frame.get("timestamp"),
        }
        outputs.append(payload)

    _write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} UDV records to {args.output}")


if __name__ == "__main__":
    main()
