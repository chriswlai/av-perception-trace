import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

try:
    from eval.error_taxonomy import summarize_failure_taxonomy
    from eval.metrics import (
        summarize_action_distribution,
        summarize_confidence_stats,
        summarize_constraint_distribution,
        summarize_trace_coverage,
    )
except ModuleNotFoundError:
    summarize_action_distribution = None
    summarize_constraint_distribution = None
    summarize_confidence_stats = None
    summarize_trace_coverage = None
    summarize_failure_taxonomy = None

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


def _index_frames_by_sample(records: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    indexed: Dict[str, Dict[str, object]] = {}
    for record in records:
        sample_token = record.get("sample_token")
        if sample_token:
            indexed[str(sample_token)] = record
    return indexed


def _index_traces_by_sample(records: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    indexed: Dict[str, Dict[str, object]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        sample_token = None
        if isinstance(metadata, dict):
            sample_token = metadata.get("sample_token")
        if sample_token:
            indexed[str(sample_token)] = record
    return indexed


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_rows(
    frames: Dict[str, Dict[str, object]],
    traces: Dict[str, Dict[str, object]],
    limit: int,
    base_dir: Path,
    overlay_paths: Dict[str, Path],
    udv_records: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[str]:
    rows: List[str] = []
    count = 0
    for sample_token, frame in frames.items():
        if limit and count >= limit:
            break
        trace = traces.get(sample_token)
        if not trace:
            continue
        udv_record = udv_records.get(sample_token) if udv_records else None
        image_paths = frame.get("image_paths") or []
        image_path = None
        if sample_token in overlay_paths:
            image_path = str(overlay_paths[sample_token])
        elif image_paths:
            image_path = image_paths[0]
        explanation = trace.get("explanation", "")
        trace_json = json.dumps(trace, indent=2, sort_keys=True)
        udv_json = json.dumps(udv_record, indent=2, sort_keys=True) if udv_record else ""
        objects = frame.get("objects") or []
        ground_truth = _ground_truth_summary(objects)
        trace_summary = _trace_summary(trace.get("relations") or [], trace.get("constraints") or [], trace.get("targets") or [])
        map_summary = _map_context_summary(frame.get("metadata") or {})
        can_bus_summary = _can_bus_summary(frame)
        action = trace.get("action", {}).get("type", "")
        udv_summary = _udv_summary(udv_record) if udv_record else ""
        rows.append(
            "\n".join(
                [
                    "<div class=\"card\">",
                    f"<div class=\"meta\">Action: {_html_escape(str(action))}</div>",
                    f"<div class=\"meta\">Sample: {_html_escape(sample_token)}</div>",
                    "<div class=\"content\">",
                    f"<div class=\"image\">{_render_image(image_path, base_dir)}</div>",
                    f"<div class=\"text\"><div class=\"gt\">{_html_escape(ground_truth)}</div><div class=\"gt\">{_html_escape(map_summary)}</div><div class=\"gt\">{_html_escape(can_bus_summary)}</div><div class=\"gt\">{_html_escape(trace_summary)}</div>{_html_escape(str(explanation))}{_render_udv_block(udv_summary, udv_json)}<pre class=\"trace\">{_html_escape(trace_json)}</pre></div>",
                    "</div>",
                    "</div>",
                ]
            )
        )
        count += 1
    return rows


def _build_summary(
    frames: Dict[str, Dict[str, object]],
    traces: Dict[str, Dict[str, object]],
) -> str:
    if not traces:
        return ""
    trace_list = list(traces.values())
    action_dist = (
        summarize_action_distribution(trace_list) if summarize_action_distribution else {}
    )
    constraint_dist = (
        summarize_constraint_distribution(trace_list) if summarize_constraint_distribution else {}
    )
    coverage = summarize_trace_coverage(trace_list) if summarize_trace_coverage else {}
    confidence = summarize_confidence_stats(trace_list) if summarize_confidence_stats else {}
    taxonomy = (
        summarize_failure_taxonomy(trace_list, frames) if summarize_failure_taxonomy else {}
    )

    action_text = ", ".join(
        f"{key}={value}" for key, value in sorted(action_dist.items())
    ) or "n/a"
    top_constraints = sorted(
        constraint_dist.items(), key=lambda item: item[1], reverse=True
    )[:6]
    constraint_text = (
        ", ".join(f"{name}={count}" for name, count in top_constraints) or "n/a"
    )
    coverage_text = (
        f"constraints={coverage.get('with_constraints', 0)}, "
        f"targets={coverage.get('with_targets', 0)}, "
        f"relations={coverage.get('with_relations', 0)}"
    )
    confidence_text = (
        f"avg={confidence.get('avg', 0.0):.2f}, "
        f"min={confidence.get('min', 0.0):.2f}, "
        f"max={confidence.get('max', 0.0):.2f}"
    )
    taxonomy_text = ", ".join(
        f"{key}={value}" for key, value in sorted(taxonomy.items())
    ) or "n/a"

    return "\n".join(
        [
            "<div class=\"summary\">",
            "<h2>Report Summary</h2>",
            "<ul>",
            f"<li>Frames: {len(frames)} | Traces: {len(traces)}</li>",
            f"<li>Action distribution: {action_text}</li>",
            f"<li>Top constraints: {constraint_text}</li>",
            f"<li>TRACE coverage: {coverage_text}</li>",
            f"<li>Confidence: {confidence_text}</li>",
            f"<li>Failure taxonomy: {taxonomy_text}</li>",
            "</ul>",
            "</div>",
        ]
    )


def _render_image(image_path: Optional[str], base_dir: Path) -> str:
    if not image_path:
        return "<div class=\"missing\">No image path</div>"
    path_value = Path(image_path)
    if not path_value.is_absolute():
        path_value = Path.cwd() / path_value
    absolute_path = path_value.resolve()
    file_uri = absolute_path.as_uri()
    try:
        relative_path = absolute_path.relative_to(base_dir)
    except ValueError:
        relative_path = absolute_path
    return (
        "<img "
        f"src=\"{_html_escape(str(relative_path))}\" "
        f"data-file=\"{_html_escape(file_uri)}\" "
        "alt=\"frame\" "
        "onerror=\"this.onerror=null;this.src=this.dataset.file;\" "
        "/>"
    )


def _ground_truth_summary(objects: List[Dict[str, object]]) -> str:
    if not objects:
        return "Detected: none."
    labels = [str(obj.get("label", "unknown")) for obj in objects]
    counts = Counter(labels)
    label_text = ", ".join(f"{label} ({count})" for label, count in counts.items())
    in_corridor = sum(bool(obj.get("in_ego_corridor")) for obj in objects)
    near_crosswalk = sum(bool(obj.get("near_crosswalk")) for obj in objects)
    distances = [
        obj.get("distance_m")
        for obj in objects
        if isinstance(obj.get("distance_m"), (int, float))
    ]
    closest = f"{min(distances):.1f}m" if distances else "n/a"
    approach_rates = [
        obj.get("approach_rate_mps")
        for obj in objects
        if isinstance(obj.get("approach_rate_mps"), (int, float))
    ]
    fastest_approach = f"{max(approach_rates):.1f}m/s" if approach_rates else "n/a"
    velocities = [
        obj.get("velocity_mps")
        for obj in objects
        if isinstance(obj.get("velocity_mps"), (int, float))
    ]
    max_velocity = f"{max(velocities):.1f}m/s" if velocities else "n/a"
    visibility = _collect_attribute_values(objects, "visibility_token")
    attribute_names = _collect_attribute_values(objects, "attribute_names")
    lidar_pts = _collect_attribute_values(objects, "num_lidar_pts")
    radar_pts = _collect_attribute_values(objects, "num_radar_pts")
    return (
        f"Detected: {label_text}. "
        f"Flags: in_ego_corridor={in_corridor}, near_crosswalk={near_crosswalk}. "
        f"Closest distance: {closest}. "
        f"Max velocity: {max_velocity}. Max approach: {fastest_approach}. "
        f"Visibility tokens: {visibility}. Attributes: {attribute_names}. "
        f"Lidar pts: {lidar_pts}. Radar pts: {radar_pts}."
    )


def _trace_summary(
    relations: List[Dict[str, object]],
    constraints: List[str],
    targets: List[Dict[str, object]],
) -> str:
    relation_text = "none"
    if relations:
        relation_text = ", ".join(
            f"{r.get('relation')}({r.get('subject_id')})" for r in relations
        )
    constraint_text = ", ".join(constraints) if constraints else "none"
    target_text = ", ".join(_format_target_summary(t) for t in targets) if targets else "none"
    return (
        f"TRACE: targets={target_text}; relations={relation_text}; constraints={constraint_text}."
    )


def _map_context_summary(metadata: Dict[str, object]) -> str:
    map_context = metadata.get("map_context")
    if not isinstance(map_context, dict):
        return "Map: n/a"
    crosswalks = map_context.get("crosswalk_tokens", [])
    lanes = map_context.get("lane_tokens", [])
    lights = map_context.get("traffic_light_tokens", [])
    stops = map_context.get("stop_line_tokens", [])
    map_name = map_context.get("map_name", "unknown")
    ego_pose = metadata.get("ego_pose")
    ego_text = "ego_pose=n/a"
    if isinstance(ego_pose, dict):
        translation = ego_pose.get("translation")
        yaw = ego_pose.get("yaw")
        ts = ego_pose.get("timestamp")
        if isinstance(translation, list) and len(translation) >= 2:
            ego_text = f"ego_pose=({translation[0]:.1f},{translation[1]:.1f}) yaw={yaw:.2f} ts={ts}"
    prev_token = metadata.get("sample_prev")
    next_token = metadata.get("sample_next")
    seq_text = f"prev={prev_token} next={next_token}"
    return (
        f"Map: {map_name}; crosswalks={len(crosswalks)}; "
        f"lanes={len(lanes)}; traffic_lights={len(lights)}; stop_lines={len(stops)}; "
        f"{ego_text}; {seq_text}."
    )


def _can_bus_summary(frame: Dict[str, object]) -> str:
    can_bus = frame.get("can_bus") or {}
    if not can_bus:
        metadata = frame.get("metadata") or {}
        if isinstance(metadata, dict):
            can_bus = metadata.get("can_bus") or {}
    if not isinstance(can_bus, dict) or not can_bus:
        return "CAN bus: n/a"
    speed = can_bus.get("vehicle_speed_mps")
    if speed is None:
        speed = can_bus.get("vehicle_speed")
    accel = can_bus.get("accel_mps2")
    brake = can_bus.get("brake")
    throttle = can_bus.get("throttle")
    state = can_bus.get("motion_state", "unknown")
    parts = []
    if isinstance(speed, (int, float)):
        parts.append(f"speed={speed:.2f}m/s")
    if isinstance(accel, (int, float)):
        parts.append(f"accel={accel:.2f}m/s^2")
    if isinstance(brake, (int, float)):
        parts.append(f"brake={brake}")
    if isinstance(throttle, (int, float)):
        parts.append(f"throttle={throttle}")
    parts.append(f"state={state}")
    return "CAN bus: " + ", ".join(parts) + "."


def _udv_summary(udv_record: Dict[str, object]) -> str:
    decide = udv_record.get("decide", {}) if isinstance(udv_record, dict) else {}
    if not isinstance(decide, dict):
        return "UDV: n/a"
    action = decide.get("action", "n/a")
    confidence = decide.get("confidence")
    constraints = decide.get("constraints", [])
    constraint_text = ", ".join(str(item) for item in constraints) if constraints else "none"
    verify_score = udv_record.get("verify_score")
    verify_text = ""
    if isinstance(verify_score, (int, float)):
        verify_text = f" verify_score={verify_score:.2f}"
    if isinstance(confidence, (int, float)):
        return (
            f"UDV: action={action} confidence={confidence:.2f} "
            f"constraints={constraint_text}{verify_text}."
        )
    return f"UDV: action={action} constraints={constraint_text}{verify_text}."


def _render_udv_block(summary: str, udv_json: str) -> str:
    if not summary:
        return ""
    if udv_json:
        return (
            f"<div class=\"gt\">{_html_escape(summary)}</div>"
            f"<pre class=\"udv\">{_html_escape(udv_json)}</pre>"
        )
    return f"<div class=\"gt\">{_html_escape(summary)}</div>"


def _format_target_summary(target: Dict[str, object]) -> str:
    label = target.get("label") or target.get("category_name") or target.get("id")
    distance = target.get("distance_m")
    velocity = target.get("velocity_mps")
    approach_rate = target.get("approach_rate_mps")
    time_delta = target.get("time_delta_s")
    yaw = target.get("yaw")
    speed_class = target.get("speed_class")
    attrs = target.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    visibility = attrs.get("visibility_token")
    lidar_pts = attrs.get("num_lidar_pts")
    radar_pts = attrs.get("num_radar_pts")
    extra = []
    if visibility is not None:
        extra.append(f"vis={visibility}")
    if lidar_pts is not None:
        extra.append(f"lidar={lidar_pts}")
    if radar_pts is not None:
        extra.append(f"radar={radar_pts}")
    if isinstance(velocity, (int, float)):
        extra.append(f"vel={velocity:.1f}m/s")
    if isinstance(approach_rate, (int, float)):
        extra.append(f"approach={approach_rate:.1f}m/s")
    if isinstance(time_delta, (int, float)):
        extra.append(f"dt={time_delta:.2f}s")
    if isinstance(yaw, (int, float)):
        extra.append(f"yaw={yaw:.2f}")
    if speed_class:
        extra.append(f"speed={speed_class}")
    extra_text = f" [{' '.join(extra)}]" if extra else ""
    if isinstance(distance, (int, float)):
        return f"{label} ({distance:.1f}m){extra_text}"
    return f"{label}{extra_text}"


def _collect_attribute_values(objects: List[Dict[str, object]], key: str) -> str:
    values = []
    for obj in objects:
        attrs = obj.get("attributes")
        if isinstance(attrs, dict) and key in attrs:
            value = attrs.get(key)
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
    if not values:
        return "n/a"
    return ", ".join(str(v) for v in values)


def _resolve_nuscenes_root(dataset_root: str, version: str) -> Path:
    root = Path(dataset_root)
    version_path = root / version
    metadata_path = version_path / "category.json"
    if metadata_path.exists():
        return root
    nested_metadata = root / version / version / "category.json"
    if nested_metadata.exists():
        return root / version
    raise FileNotFoundError(
        f"Missing nuScenes metadata at {metadata_path}. "
        "Ensure dataset_root points to the folder containing the version directory."
    )


def _render_overlays(
    sample_tokens: List[str],
    dataset_root: str,
    version: str,
    output_dir: Path,
) -> Dict[str, Path]:
    try:
        from nuscenes.nuscenes import NuScenes  # type: ignore
    except ModuleNotFoundError:
        return {}

    resolved_root = _resolve_nuscenes_root(dataset_root, version)
    nusc = NuScenes(version=version, dataroot=str(resolved_root), verbose=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_paths: Dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        plt = None
    for token in sample_tokens:
        sample = nusc.get("sample", token)
        cam_token = sample["data"].get("CAM_FRONT")
        if not cam_token:
            continue
        out_path = output_dir / f"{token}.jpg"
        nusc.render_sample_data(
            cam_token,
            out_path=str(out_path),
            with_anns=True,
            verbose=False,
        )
        if plt is not None:
            plt.close("all")
        overlay_paths[token] = out_path
    return overlay_paths


def _build_html(rows: List[str]) -> str:
    summary = rows[0] if rows and rows[0].startswith("<div class=\"summary\">") else ""
    remaining_rows = rows[1:] if summary else rows
    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset=\"utf-8\"/>",
            "<title>TRACE Sanity Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; }",
            ".card { background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }",
            ".summary { background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }",
            ".summary h2 { margin-top: 0; font-size: 18px; }",
            ".summary ul { margin: 0; padding-left: 20px; }",
            ".content { display: flex; gap: 16px; align-items: flex-start; }",
            ".image img { max-width: 420px; border-radius: 6px; }",
            ".text { font-size: 14px; line-height: 1.4; }",
            ".gt { font-size: 12px; color: #222; margin-bottom: 8px; }",
            ".trace { background: #f2f3f5; padding: 8px; border-radius: 6px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }",
            ".udv { background: #eef6ff; padding: 8px; border-radius: 6px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }",
            ".meta { font-size: 12px; color: #555; margin-bottom: 4px; }",
            ".missing { color: #b00; font-size: 12px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>TRACE Sanity Report</h1>",
            summary,
            *remaining_rows,
            "</body>",
            "</html>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="data/perception_frames.jsonl")
    parser.add_argument("--traces", default="data/teacher_traces.jsonl")
    parser.add_argument("--output", default="data/report.html")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--use-factor-traces", action="store_true")
    parser.add_argument("--udv", default="")
    args = parser.parse_args()

    traces_path = args.traces
    if args.use_factor_traces:
        traces_path = "data/factor_traces.jsonl"

    frames = _index_frames_by_sample(_load_jsonl(args.frames))
    traces = _index_traces_by_sample(_load_jsonl(traces_path))
    udv_records: Optional[Dict[str, Dict[str, object]]] = None
    if args.udv:
        udv_records = _index_traces_by_sample(_load_jsonl(args.udv))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_paths: Dict[str, Path] = {}
    if args.overlay:
        overlay_paths = _render_overlays(
            list(frames.keys())[: args.limit],
            dataset_root=args.dataset_root,
            version=args.version,
            output_dir=output_path.parent / "report_images",
        )
    rows = _build_rows(
        frames,
        traces,
        args.limit,
        output_path.parent.resolve(),
        overlay_paths,
        udv_records,
    )
    summary = _build_summary(frames, traces)
    if summary:
        rows = [summary, *rows]
    html = _build_html(rows)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
