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

## Quickstart (scaffold)
Install dependencies:
```
pip install -r requirements.txt
```

Notes:
- The TRACE/UDV structure lives in `src/` as outlined above.

## Next steps
- Implement `src/data/schemas.py` validation checks and a minimal unit test.
- Connect nuScenes mini loader to `src/perception/features.py`.
- Build teacher traces via `src/teacher/teacher_runner.py`.
- Add evaluation summaries in `src/eval/metrics.py`.

## Dummy pipeline (current)
Run the dummy inference and teacher/eval flow:
```
python scripts/run_infer.py
python scripts/run_teacher.py
python scripts/run_eval.py
```

`run_infer.py` writes `data/perception_frames.jsonl`, which `run_teacher.py` reads by default.

## nuScenes mini usage
If you unzipped the dataset into `data/v1.0-mini/` (so the JSON metadata lives at
`data/v1.0-mini/v1.0-mini/*.json`), run:
```
python scripts/run_infer.py --dataset-root data/v1.0-mini --limit 50
```

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
