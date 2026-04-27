# AV Perception TRACE

An observable perception–reasoning system for autonomous driving that produces
structured decision traces and interpretable explanations.

## Overview

Modern autonomous driving systems can predict actions, but often lack transparency in
why those actions were taken.

This project introduces a structured protocol for perception-driven reasoning:

- **TRACE** — Targets, Relations, Action, Constraints, Explanation
- **UDV** — Understand → Decide → Verify reasoning loop

The system explicitly represents:
- what the vehicle perceives,
- which risks matter,
- how decisions are made,
- and how those decisions can be verified.

## Key Idea

Instead of:

```
image → black-box model → action
```

We build:

```
image → perception signals → structured reasoning (TRACE) → decision → explanation
```

Explanations are derived from structured reasoning, not generated post-hoc.

## High-Level System

```mermaid
flowchart LR
    A[Visual Input] --> B[Explicit Perception Signals]
    B --> C[TRACE Protocol]
    C --> D[Decision + Explanation]
    C --> E[Understand-Decide-Verify Reasoner]
    E --> F[Verified Reasoning Output]
    D --> G[Evaluation + Report]
    F --> G
```

TRACE acts as the observability contract between perception, learned reasoning,
decision-making, and explanation generation.

## Example Output (TRACE)

```json
{
  "targets": ["pedestrian_3"],
  "relations": ["in_ego_corridor"],
  "action": {"type": "STOP", "confidence": 0.9},
  "constraints": ["pedestrian_in_path"],
  "explanation": "The vehicle stopped because a pedestrian was detected in the ego corridor."
}
```

## Example Output (UDV)

```json
{
  "understand": {
    "salient_objects": ["pedestrian_3"],
    "risks": ["pedestrian_crossing"],
    "uncertainty": "low"
  },
  "decide": {
    "action": "STOP",
    "confidence": 0.88
  },
  "verify": {
    "checks": ["pedestrian_in_path → STOP"],
    "counterfactuals": [
      "If no pedestrian were present, action would be PROCEED"
    ]
  }
}
```

## System Structure

```
src/
  data/            # schemas + nuScenes loader
  perception/      # feature extraction + uncertainty
  trace_protocol/  # TRACE types, builders, validators
  teacher/         # deterministic rules (ground truth reasoning)
  models/          # learned factor model
  udv/             # UDV reasoning + verification
  eval/            # metrics + failure taxonomy
scripts/           # runnable pipelines
docs/              # design documentation
```

## Pipeline

One-shot pipeline (recommended):
```
python scripts/run_all.py \
  --dataset-root data/v1.0-mini \
  --limit 120 \
  --report-limit 120
```

This runs:
- perception frame generation
- TRACE generation (teacher rules)
- evaluation summary
- HTML report generation

Optional: include UDV reasoning:
```
python scripts/run_all.py --run-udv --udv-output data/udv_outputs.jsonl
```

## Architecture

```mermaid
flowchart LR
    A[Camera Frame / nuScenes Sample] --> B[Perception Layer]
    B --> B1[Objects<br/>pedestrians, vehicles, cyclists]
    B --> B2[Map Context<br/>lanes, crosswalks, stop lines]
    B --> B3[Motion / Uncertainty<br/>distance, velocity, approach rate]
    B1 --> C[PerceptionFrame]
    B2 --> C
    B3 --> C
    C --> D[TRACE Teacher Rules]
    D --> E[TRACE Record]
    E --> E1[Targets]
    E --> E2[Relations]
    E --> E3[Action]
    E --> E4[Constraints]
    E --> E5[Explanation]
    C --> F[Learned Factor Model]
    F --> G[Predicted TRACE]
    C --> H[UDV Reasoner]
    H --> H1[Understand]
    H1 --> H2[Decide]
    H2 --> H3[Verify]
    H3 --> I[UDV Output]
    E --> J[Evaluation + Failure Taxonomy]
    G --> J
    I --> J
    J --> K[HTML Report / Debug View]
```

## Outputs

TRACE logs:
- structured reasoning for every frame
- fully auditable decision pipeline

HTML report:
- visual + structured debugging interface
- action, perception context, constraints, explanations

Evaluation summary:
- action distribution
- constraint frequency
- failure taxonomy
- confidence statistics

## Individual Components

Generate perception frames:
```
PYTHONPATH=src python scripts/run_infer.py \
  --dataset-root data/v1.0-mini \
  --limit 50 \
  --output data/perception_frames.jsonl
```

Generate TRACE (teacher rules):
```
PYTHONPATH=src python scripts/run_teacher.py \
  --input data/perception_frames.jsonl \
  --output data/teacher_traces.jsonl \
  --limit 50
```

Evaluate:
```
PYTHONPATH=src python scripts/run_eval.py \
  --traces data/teacher_traces.jsonl
```

Render report:
```
PYTHONPATH=src python scripts/render_report.py \
  --frames data/perception_frames.jsonl \
  --traces data/teacher_traces.jsonl \
  --output data/report.html
```

Learned reasoning (factor model) training:
```
PYTHONPATH=src python scripts/run_train_factors.py
```

Learned reasoning (factor model) inference:
```
PYTHONPATH=src python src/models/infer.py \
  --frames data/perception_frames.jsonl \
  --model data/factor_model.pkl \
  --output data/factor_traces.jsonl
```

Evaluate vs teacher:
```
PYTHONPATH=src python scripts/evaluate_factor.py \
  --teacher data/teacher_traces.jsonl \
  --factor data/factor_traces.jsonl
```

## Report Interpretation Guide

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

## Design Goals

- make perception explicit and structured
- ensure reasoning is observable and auditable
- support failure analysis, not just prediction
- bridge deterministic rules → learned reasoning → LLM-based reasoning

## Future Work

- lightweight LLM for structured UDV reasoning
- temporal reasoning (multi-frame context)
- counterfactual reasoning for safety validation
- improved uncertainty modeling

## Summary

This project treats perception as a contract between sensing and decision-making. By
enforcing structured reasoning (TRACE) and verifiable decision loops (UDV), it
demonstrates how autonomous systems can remain interpretable even when learning-based
components are introduced.

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
