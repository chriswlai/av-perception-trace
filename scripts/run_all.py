import subprocess
import sys


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    _run([sys.executable, "scripts/run_infer.py"])
    _run([sys.executable, "scripts/run_teacher.py"])
    _run([sys.executable, "scripts/run_eval.py"])


if __name__ == "__main__":
    main()
