import json
from pathlib import Path

from data.schemas import validate_perception_frame
from udv.udv_engine import run_udv_reasoner
from udv.udv_verify import score_udv_record


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def test_udv_regression_set() -> None:
    fixtures_dir = Path("tests/fixtures")
    frames = _load_jsonl(fixtures_dir / "udv_frames.jsonl")
    expected = _load_jsonl(fixtures_dir / "udv_expected.jsonl")

    expected_by_token = {
        record.get("metadata", {}).get("sample_token"): record for record in expected
    }

    for frame in frames:
        sample_token = frame.get("sample_token")
        assert sample_token in expected_by_token
        expected_record = expected_by_token[sample_token]

        frame_model = validate_perception_frame(frame)
        udv_record = run_udv_reasoner(frame)
        verify_score = score_udv_record(frame_model, udv_record)

        assert udv_record.decide.action.value == expected_record["decide"]["action"]
        assert set(udv_record.decide.constraints) == set(
            expected_record["decide"]["constraints"]
        )
        assert verify_score == expected_record["verify_score"]
