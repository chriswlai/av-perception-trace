import argparse
import json
from pathlib import Path
from typing import List

from data.schemas import TraceRecord
from eval.metrics import summarize_action_distribution


def _load_traces(path: str | Path) -> List[TraceRecord]:
    traces: List[TraceRecord] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            traces.append(TraceRecord.model_validate(payload))
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/teacher_traces.jsonl")
    args = parser.parse_args()

    traces = _load_traces(args.traces)
    action_counts = summarize_action_distribution(traces)
    print("Action distribution:", action_counts)


if __name__ == "__main__":
    main()
