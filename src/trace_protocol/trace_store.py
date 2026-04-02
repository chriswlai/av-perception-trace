import json
from pathlib import Path
from typing import Iterable

from data.schemas import TraceRecord


def write_traces_jsonl(path: str | Path, traces: Iterable[TraceRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(trace.model_dump_json())
            handle.write("\n")
