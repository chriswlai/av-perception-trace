import argparse
import os
import subprocess
import sys
from pathlib import Path


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(Path("src").resolve())]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, env=_python_env())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/v1.0-mini")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--frames", default="data/perception_frames.jsonl")
    parser.add_argument("--traces", default="data/teacher_traces.jsonl")
    parser.add_argument("--report", default="data/report.html")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--report-limit", type=int, default=0)
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--udv", default="")
    parser.add_argument("--run-udv", action="store_true")
    parser.add_argument("--udv-output", default="data/udv_outputs.jsonl")
    parser.add_argument("--udv-limit", type=int, default=0)
    args = parser.parse_args()

    frames_path = Path(args.frames)
    traces_path = Path(args.traces)

    if not args.skip_infer:
        _run(
            [
                sys.executable,
                "scripts/run_infer.py",
                "--dataset-root",
                args.dataset_root,
                "--limit",
                str(args.limit),
                "--output",
                args.frames,
            ]
        )
    elif not frames_path.exists():
        raise SystemExit(f"Frames file not found: {frames_path}")

    frame_count = _count_lines(frames_path)
    if frame_count <= 0:
        raise SystemExit(f"No frames found in {frames_path}")

    _run(
        [
            sys.executable,
            "scripts/run_teacher.py",
            "--input",
            args.frames,
            "--output",
            args.traces,
            "--limit",
            str(frame_count),
        ]
    )

    _run([sys.executable, "scripts/run_eval.py", "--traces", args.traces])

    udv_path = args.udv
    if args.run_udv:
        udv_limit = args.udv_limit if args.udv_limit > 0 else frame_count
        _run(
            [
                sys.executable,
                "scripts/run_udv_llm.py",
                "--frames",
                args.frames,
                "--output",
                args.udv_output,
                "--limit",
                str(udv_limit),
            ]
        )
        udv_path = args.udv_output

    render_cmd = [
        sys.executable,
        "scripts/render_report.py",
        "--frames",
        args.frames,
        "--traces",
        args.traces,
        "--output",
        args.report,
        "--limit",
        str(args.report_limit),
        "--dataset-root",
        args.dataset_root,
        "--version",
        args.version,
    ]
    if args.overlay:
        render_cmd.append("--overlay")
    if udv_path:
        render_cmd.extend(["--udv", udv_path])
    _run(render_cmd)


if __name__ == "__main__":
    main()
