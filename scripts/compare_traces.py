import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


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


def _action_type(record: Dict[str, object]) -> str:
    action = record.get("action", {})
    if isinstance(action, dict):
        action_type = action.get("type")
        return str(action_type) if action_type else ""
    return ""


def _constraints(record: Dict[str, object]) -> List[str]:
    value = record.get("constraints", [])
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _constraint_scores(pred: List[str], truth: List[str]) -> Tuple[float, float, float]:
    pred_set = set(pred)
    truth_set = set(truth)
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="data/teacher_traces.jsonl")
    parser.add_argument("--factor", default="data/factor_traces.jsonl")
    args = parser.parse_args()

    teacher_records = _load_jsonl(args.teacher)
    factor_records = _load_jsonl(args.factor)
    if not teacher_records or not factor_records:
        raise RuntimeError("Missing teacher or factor traces.")

    teacher_by_sample = _index_traces(teacher_records)
    factor_by_sample = _index_traces(factor_records)
    common_tokens = sorted(set(teacher_by_sample) & set(factor_by_sample))
    if not common_tokens:
        raise RuntimeError("No overlapping sample tokens between traces.")

    action_matches = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for token in common_tokens:
        teacher = teacher_by_sample[token]
        factor = factor_by_sample[token]
        if _action_type(teacher) == _action_type(factor):
            action_matches += 1
        precision, recall, f1 = _constraint_scores(
            _constraints(factor), _constraints(teacher)
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    total = len(common_tokens)
    print("Trace comparison summary")
    print(f"Samples compared: {total}")
    print(f"Action accuracy: {action_matches / total:.2%}")
    print(f"Constraint precision: {precision_sum / total:.2%}")
    print(f"Constraint recall: {recall_sum / total:.2%}")
    print(f"Constraint F1: {f1_sum / total:.2%}")


if __name__ == "__main__":
    main()
