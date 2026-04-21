# AV UDV TRACE Protocol

This project implements an observable perception–reasoning protocol for autonomous
driving, centered on TRACE and an Understand–Decide–Verify (UDV) reasoning model.
Explanations are derived from structured TRACE outputs, not post-hoc text.

## Structure (V2.1)
- `src/data/`: dataset schemas + nuScenes loader
- `src/perception/`: feature extraction and uncertainty
- `src/trace_protocol/`: TRACE types, builders, validators, storage
- `src/teacher/`: deterministic rules for TRACE supervision
- `src/models/`: learned factor model stubs
- `src/udv/`: UDV schema, prompts, verification
- `src/eval/`: metrics and error taxonomy
- `scripts/`: runnable entry points
- `docs/`: design documentation

## Quickstart
Install dependencies:
```
pip install -r requirements.txt
```

Notes:
- The TRACE/UDV code lives in `src/`.
- Most scripts assume `PYTHONPATH=src` (the one-shot script handles this for you).

## One-shot flow (recommended)
Generate frames → teacher traces → eval summary → report in one command:
```
python scripts/run_all.py --dataset-root data/v1.0-mini --limit 120 --report-limit 120
```

Use existing frames (e.g., `data/diverse_frames.jsonl`):
```
python scripts/run_all.py --skip-infer --frames data/diverse_frames.jsonl \
  --traces data/diverse_traces.jsonl --report data/diverse_report.html --report-limit 120
```

Include UDV outputs in the report:
```
python scripts/run_all.py --run-udv --udv-output data/udv_outputs.jsonl
```

Common flags:
- `--dataset-root`: nuScenes root (default `data/v1.0-mini`)
- `--limit`: number of frames to generate
- `--report-limit`: number of report cards (0 = all)
- `--overlay`: render nuScenes overlays into `data/report_images/`
- `--run-udv`: generate UDV outputs and inject them into report
- `--udv-limit`: number of frames for UDV (0 = all)

## Individual executables
Generate perception frames (nuScenes mini):
```
PYTHONPATH=src python scripts/run_infer.py --dataset-root data/v1.0-mini --limit 50 --output data/perception_frames.jsonl
```

Generate teacher traces:
```
PYTHONPATH=src python scripts/run_teacher.py --input data/perception_frames.jsonl --output data/teacher_traces.jsonl --limit 50
```

Summarize action distribution:
```
PYTHONPATH=src python scripts/run_eval.py --traces data/teacher_traces.jsonl
```

Render report:
```
PYTHONPATH=src python scripts/render_report.py --frames data/perception_frames.jsonl --traces data/teacher_traces.jsonl --output data/report.html --limit 50
```

Render report with overlays:
```
PYTHONPATH=src python scripts/render_report.py --frames data/perception_frames.jsonl --traces data/teacher_traces.jsonl --output data/report.html --limit 50 --overlay --dataset-root data
```

Generate UDV outputs:
```
PYTHONPATH=src python scripts/run_udv_llm.py --frames data/perception_frames.jsonl --output data/udv_outputs.jsonl --limit 50
```

Train and infer the factor model:
```
PYTHONPATH=src python scripts/run_train_factors.py
PYTHONPATH=src python scripts/run_infer.py --dataset-root data/v1.0-mini --limit 120 --output data/perception_frames.jsonl
PYTHONPATH=src python src/models/infer.py --frames data/perception_frames.jsonl --model data/factor_model.pkl --output data/factor_traces.jsonl
```

Evaluate factor model vs teacher traces:
```
PYTHONPATH=src python scripts/evaluate_factor.py --teacher data/teacher_traces.jsonl --factor data/factor_traces.jsonl
```

## Report interpretation guide
Each card corresponds to one frame:
- **Action**: TRACE action chosen by deterministic teacher rules.
- **Detected**: object counts and context (closest distance, velocity, approach rate).
- **Map**: lanes/crosswalks/stop lines and ego pose, if available.
- **CAN bus**: speed/accel/brake/throttle and derived `motion_state`.
- **TRACE**: targets/relations/constraints that justified the action.
- **UDV** (optional): deterministic UDV output with `verify_score`.

Summary section:
- **Action distribution**: counts of STOP/SLOW/PROCEED.
- **Top constraints**: most frequent constraint types.
- **TRACE coverage**: percent of traces with targets/relations/constraints.
- **Confidence**: min/avg/max action confidence.
- **Failure taxonomy**: heuristic error tags for quick review.

## Heuristic perception flags (current)
The loader assigns simple distance-based flags for pedestrians and cyclists:
- Label mapping: `human.pedestrian.*` -> `pedestrian`, `vehicle.bicycle`/`vehicle.motorcycle` -> `cyclist`
- `in_ego_corridor`: distance <= 12.0 m
- `near_crosswalk`: distance <= 20.0 m

Or run them all at once:
```
python scripts/run_all.py
```

## Sanity check report
Render a simple HTML page with images + explanations:
```
python scripts/render_report.py --limit 20
```
Open `data/report.html` in your browser.

To render images with nuScenes 3D boxes overlaid:
```
python scripts/render_report.py --limit 20 --overlay --dataset-root data
```
