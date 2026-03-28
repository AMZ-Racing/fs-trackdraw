from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import yaml

from .configuration import dump_rule_settings
from .models import BackgroundSpec, GeneratorSettings, ProjectState, RuleSettings, TrackOverlay


def save_project(filename: str, project_state: ProjectState) -> None:
    payload = {
        "version": 2,
        "background": asdict(project_state.background),
        "allowed_area": {"points": project_state.allowed_area_points},
        "track_mode": project_state.track_mode,
        "manual_centerline_control_points": project_state.control_points,
        "generator_settings": asdict(project_state.generator_settings),
        "track_width_profile": {"control_points": project_state.width_control_points},
        "cone_spacing_profile": {"control_points": project_state.cone_spacing_control_points},
        "cone_overrides": {
            "left": project_state.left_cone_overrides,
            "right": project_state.right_cone_overrides,
        },
        "imported_track": asdict(project_state.imported_track),
        "validation_rule_overrides": dump_rule_settings(project_state.rules),
    }
    with open(filename, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_project(filename: str) -> ProjectState:
    with open(filename, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    background_data = payload.get("background", {})
    generator_data = payload.get("generator_settings", {})
    imported_track_data = payload.get("imported_track", {})
    rule_data: Dict[str, Dict[str, object]] = payload.get("validation_rule_overrides", {})

    rules = {
        rule_id: RuleSettings(
            enabled=bool(values.get("enabled", True)),
            threshold=float(values["threshold"]) if values.get("threshold") is not None else None,
            severity=str(values.get("severity", "error")),
        )
        for rule_id, values in rule_data.items()
    }

    state = ProjectState(
        background=BackgroundSpec(**background_data) if background_data else BackgroundSpec(),
        allowed_area_points=[tuple(point) for point in payload.get("allowed_area", {}).get("points", [])],
        control_points=[tuple(point) for point in payload.get("manual_centerline_control_points", [])],
        track_mode=str(payload.get("track_mode", "manual")),
        generator_settings=GeneratorSettings(**generator_data) if generator_data else GeneratorSettings(),
        rules=rules,
        width_control_points=[tuple(point) for point in payload.get("track_width_profile", {}).get("control_points", [])],
        cone_spacing_control_points=[tuple(point) for point in payload.get("cone_spacing_profile", {}).get("control_points", [])],
        left_cone_overrides=[tuple(point) for point in payload.get("cone_overrides", {}).get("left", [])],
        right_cone_overrides=[tuple(point) for point in payload.get("cone_overrides", {}).get("right", [])],
        imported_track=TrackOverlay(
            source_path=str(imported_track_data.get("source_path", "")),
            left_cones_local_m=[tuple(point) for point in imported_track_data.get("left_cones_local_m", [])],
            right_cones_local_m=[tuple(point) for point in imported_track_data.get("right_cones_local_m", [])],
            control_points_local_m=[tuple(point) for point in imported_track_data.get("control_points_local_m", [])],
            center_x_px=float(imported_track_data.get("center_x_px", 0.0)),
            center_y_px=float(imported_track_data.get("center_y_px", 0.0)),
            rotation_deg=float(imported_track_data.get("rotation_deg", 0.0)),
            locked=bool(imported_track_data.get("locked", False)),
            preserve_cones=bool(imported_track_data.get("preserve_cones", False)),
        ),
        project_path=filename,
    )
    return state
