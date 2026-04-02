import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib

from data.schemas import ActionType, TraceAction, TraceRecord
from models.factor_model import extract_features
from trace_protocol.render_explanation import render_explanation


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


def _write_jsonl(path: str | Path, records: List[TraceRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def run_inference() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/perception_frames.jsonl")
    parser.add_argument("--model", default="data/factor_model.pkl")
    parser.add_argument("--output", default="data/factor_traces.jsonl")
    args = parser.parse_args()

    frames = _load_jsonl(args.frames)
    if not frames:
        raise RuntimeError("Frames are empty. Generate them first.")

    model_bundle = joblib.load(args.model)
    feature_names = model_bundle["feature_names"]
    action_model = model_bundle["action_model"]
    constraint_model = model_bundle["constraint_model"]
    constraint_labels = model_bundle["constraint_labels"]

    traces: List[TraceRecord] = []
    for frame in frames:
        features = extract_features(frame)
        vector = [features.get(name, 0.0) for name in feature_names]
        action_pred = action_model.predict([vector])[0]
        action_probs = action_model.predict_proba([vector])[0]
        confidence = float(max(action_probs))
        constraint_pred = constraint_model.predict([vector])[0]
        predicted_constraints = [
            constraint_labels[idx]
            for idx, value in enumerate(constraint_pred)
            if value
        ]
        explanation = render_explanation(action_pred, predicted_constraints, [])
        traces.append(
            TraceRecord(
                targets=[],
                relations=[],
                action=TraceAction(type=ActionType(action_pred), confidence=confidence),
                constraints=predicted_constraints,
                explanation=explanation,
                can_bus=frame.get("can_bus", {}),
                metadata={
                    "scene_id": frame.get("scene_id"),
                    "sample_token": frame.get("sample_token"),
                    "timestamp": frame.get("timestamp"),
                },
            )
        )

    _write_jsonl(args.output, traces)
    print(f"Wrote {len(traces)} factor traces to {args.output}")


if __name__ == "__main__":
    run_inference()
