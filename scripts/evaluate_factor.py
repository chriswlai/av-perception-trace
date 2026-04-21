import argparse
import json
from pathlib import Path
from typing import Dict, List

from eval.metrics import compute_action_accuracy, compute_constraint_scores


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="data/teacher_traces.jsonl")
    parser.add_argument("--factor", default="data/factor_traces.jsonl")
    args = parser.parse_args()

    teacher = _load_jsonl(args.teacher)
    factor = _load_jsonl(args.factor)
    if not teacher or not factor:
        raise RuntimeError("Missing teacher or factor traces.")

    action_accuracy = compute_action_accuracy(factor, teacher)
    constraint_scores = compute_constraint_scores(factor, teacher)

    print("Phase 2 evaluation")
    print(f"Action accuracy: {action_accuracy:.2%}")
    print(
        "Constraint precision/recall/F1: "
        f"{constraint_scores['precision']:.2%} / "
        f"{constraint_scores['recall']:.2%} / "
        f"{constraint_scores['f1']:.2%}"
    )


if __name__ == "__main__":
    main()
