import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from models.factor_model import extract_features


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


def _index_traces(records: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    indexed: Dict[str, Dict[str, object]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        sample_token = metadata.get("sample_token")
        if sample_token:
            indexed[str(sample_token)] = record
    return indexed


def train_factor_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/perception_frames.jsonl")
    parser.add_argument("--traces", default="data/teacher_traces.jsonl")
    parser.add_argument("--output", default="data/factor_model.pkl")
    args = parser.parse_args()

    frames = _load_jsonl(args.frames)
    traces = _index_traces(_load_jsonl(args.traces))
    if not frames or not traces:
        raise RuntimeError("Frames or traces are empty. Generate them first.")

    feature_rows: List[Dict[str, float]] = []
    actions: List[str] = []
    constraints: List[List[str]] = []
    for frame in frames:
        sample_token = frame.get("sample_token")
        if not sample_token or str(sample_token) not in traces:
            continue
        trace = traces[str(sample_token)]
        feature_rows.append(extract_features(frame))
        action = trace.get("action", {}).get("type", "PROCEED")
        actions.append(str(action))
        constraint_list = trace.get("constraints", [])
        if isinstance(constraint_list, list):
            constraints.append([str(item) for item in constraint_list])
        else:
            constraints.append([])

    if not feature_rows:
        raise RuntimeError("No matching frames/traces to train on.")

    feature_names = sorted(feature_rows[0].keys())
    x = [[row.get(name, 0.0) for name in feature_names] for row in feature_rows]

    action_model = LogisticRegression(max_iter=200, multi_class="multinomial")
    action_model.fit(x, actions)

    mlb = MultiLabelBinarizer()
    y_constraints = mlb.fit_transform(constraints)
    constraint_model = OneVsRestClassifier(LogisticRegression(max_iter=200))
    constraint_model.fit(x, y_constraints)

    joblib.dump(
        {
            "feature_names": feature_names,
            "action_model": action_model,
            "constraint_model": constraint_model,
            "constraint_labels": list(mlb.classes_),
        },
        args.output,
    )
    print(f"Saved factor model to {args.output}")


if __name__ == "__main__":
    train_factor_model()
