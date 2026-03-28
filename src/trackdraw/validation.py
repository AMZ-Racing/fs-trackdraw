from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .geometry import closed_loop_segment_lengths, points_to_array
from .models import RuleSettings, TrackGeometry, ValidationIssue, ValidationResult


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    label: str
    threshold_label: Optional[str]
    default_threshold: Optional[float]
    default_severity: str = "error"


RULE_DEFINITIONS: Dict[str, RuleDefinition] = {
    "centerline_simple": RuleDefinition("centerline_simple", "Centerline is simple", None, None),
    "left_boundary_simple": RuleDefinition("left_boundary_simple", "Left boundary is simple", None, None),
    "right_boundary_simple": RuleDefinition("right_boundary_simple", "Right boundary is simple", None, None),
    "boundaries_do_not_cross": RuleDefinition("boundaries_do_not_cross", "Boundaries do not cross", None, None),
    "cones_inside_area": RuleDefinition("cones_inside_area", "Cones stay inside area", None, None),
    "min_clearance": RuleDefinition("min_clearance", "Minimum area clearance", "m", 1.0),
    "min_track_width": RuleDefinition("min_track_width", "Minimum track width", "m", 3.0),
    "min_centerline_radius": RuleDefinition("min_centerline_radius", "Minimum centerline radius", "m", 4.5),
    "min_cone_spacing": RuleDefinition("min_cone_spacing", "Minimum cone spacing", "m", 2.0),
    "max_cone_spacing": RuleDefinition("max_cone_spacing", "Maximum cone spacing", "m", 6.0, default_severity="warning"),
    "min_track_length": RuleDefinition("min_track_length", "Minimum track length", "m", 80.0, default_severity="warning"),
}


def build_default_rules(overrides: Dict[str, RuleSettings] | None = None) -> Dict[str, RuleSettings]:
    rules: Dict[str, RuleSettings] = {}
    overrides = overrides or {}
    for rule_id, definition in RULE_DEFINITIONS.items():
        override = overrides.get(rule_id)
        if override is not None:
            rules[rule_id] = RuleSettings(
                enabled=override.enabled,
                threshold=override.threshold if override.threshold is not None else definition.default_threshold,
                severity=override.severity,
            )
        else:
            rules[rule_id] = RuleSettings(
                enabled=True,
                threshold=definition.default_threshold,
                severity=definition.default_severity,
            )
    return rules


def make_issue(rule_id: str, severity: str, summary: str, detail: str) -> ValidationIssue:
    return ValidationIssue(rule_id=rule_id, severity=severity, summary=summary, detail=detail)


def _closed_line(points: np.ndarray) -> LineString:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return LineString()
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack((arr, arr[0]))
    return LineString(arr)


def _cone_spacing_metrics(points: np.ndarray, px_per_m: float) -> np.ndarray:
    lengths_px = closed_loop_segment_lengths(points)
    return lengths_px / float(px_per_m) if lengths_px.size else np.array([], dtype=float)


def validate_track(
    allowed_area_points: Iterable[Iterable[float]],
    geometry: TrackGeometry,
    px_per_m: float,
    rules: Dict[str, RuleSettings],
) -> ValidationResult:
    result = ValidationResult()
    allowed_area = points_to_array(allowed_area_points)
    if allowed_area.shape[0] < 3:
        result.issues.append(
            make_issue(
                "allowed_area",
                "error",
                "Allowed area is incomplete",
                "Define at least three allowed-area vertices before creating a track.",
            )
        )
        return result

    polygon = Polygon(allowed_area)
    if not polygon.is_valid or polygon.area <= 0:
        result.issues.append(
            make_issue(
                "allowed_area",
                "error",
                "Allowed area is invalid",
                "The allowed-area polygon self-intersects or has zero area.",
            )
        )
        return result

    if not geometry.has_track:
        result.issues.append(
            make_issue(
                "track_missing",
                "error",
                "Track is incomplete",
                "Add or generate enough control points to build a closed track.",
            )
        )
        return result

    centerline = np.asarray(geometry.centerline, dtype=float)
    left_boundary = np.asarray(geometry.left_boundary, dtype=float)
    right_boundary = np.asarray(geometry.right_boundary, dtype=float)
    left_cones = np.asarray(geometry.left_cones, dtype=float)
    right_cones = np.asarray(geometry.right_cones, dtype=float)

    centerline_line = _closed_line(centerline)
    left_line = _closed_line(left_boundary)
    right_line = _closed_line(right_boundary)

    min_width_m = float(np.min(np.linalg.norm(left_boundary[:-1] - right_boundary[:-1], axis=1) / px_per_m))
    min_clearance_m = float(min(left_line.distance(polygon.exterior), right_line.distance(polygon.exterior)) / px_per_m)
    left_spacing = _cone_spacing_metrics(left_cones, px_per_m)
    right_spacing = _cone_spacing_metrics(right_cones, px_per_m)
    all_spacing = np.concatenate([left_spacing, right_spacing]) if left_spacing.size or right_spacing.size else np.array([])

    result.metrics.update(
        {
            "track_length_m": float(geometry.track_length_m),
            "min_radius_m": float(geometry.min_radius_m),
            "min_track_width_m": min_width_m,
            "min_clearance_m": min_clearance_m,
            "min_cone_spacing_m": float(np.min(all_spacing)) if all_spacing.size else float("inf"),
            "max_cone_spacing_m": float(np.max(all_spacing)) if all_spacing.size else 0.0,
        }
    )

    def enabled(rule_id: str) -> bool:
        return rules.get(rule_id, RuleSettings()).enabled

    def severity(rule_id: str) -> str:
        return rules.get(rule_id, RuleSettings()).severity

    def threshold(rule_id: str, fallback: float) -> float:
        value = rules.get(rule_id, RuleSettings(threshold=fallback)).threshold
        return float(fallback if value is None else value)

    if enabled("centerline_simple") and not centerline_line.is_simple:
        result.issues.append(
            make_issue("centerline_simple", severity("centerline_simple"), "Centerline self-intersects", "Manual or generated centerline crosses itself.")
        )
    if enabled("left_boundary_simple") and not left_line.is_simple:
        result.issues.append(
            make_issue("left_boundary_simple", severity("left_boundary_simple"), "Left boundary self-intersects", "The left track boundary is not a simple loop.")
        )
    if enabled("right_boundary_simple") and not right_line.is_simple:
        result.issues.append(
            make_issue("right_boundary_simple", severity("right_boundary_simple"), "Right boundary self-intersects", "The right track boundary is not a simple loop.")
        )
    if enabled("boundaries_do_not_cross") and left_line.distance(right_line) < 1e-6:
        result.issues.append(
            make_issue("boundaries_do_not_cross", severity("boundaries_do_not_cross"), "Track boundaries cross or touch", "Left and right boundaries overlap or intersect.")
        )

    if enabled("cones_inside_area"):
        outside = 0
        boundary_outside = 0
        for point in np.vstack((left_cones, right_cones)):
            if not polygon.covers(Point(float(point[0]), float(point[1]))):
                outside += 1
        for point in np.vstack((left_boundary, right_boundary)):
            if not polygon.covers(Point(float(point[0]), float(point[1]))):
                boundary_outside += 1
        if outside or boundary_outside:
            result.issues.append(
                make_issue(
                    "cones_inside_area",
                    severity("cones_inside_area"),
                    "Geometry leaves allowed area",
                    f"{outside} cones and {boundary_outside} boundary samples fall outside the allowed area polygon.",
                )
            )

    if enabled("min_clearance"):
        clearance_threshold = threshold("min_clearance", 1.0)
        if min_clearance_m + 1e-6 < clearance_threshold:
            result.issues.append(
                make_issue(
                    "min_clearance",
                    severity("min_clearance"),
                    "Track too close to allowed-area edge",
                    f"Minimum boundary clearance is {min_clearance_m:.2f} m, below the configured {clearance_threshold:.2f} m.",
                )
            )

    if enabled("min_track_width"):
        width_threshold = threshold("min_track_width", 3.0)
        if min_width_m + 1e-6 < width_threshold:
            result.issues.append(
                make_issue(
                    "min_track_width",
                    severity("min_track_width"),
                    "Track width below minimum",
                    f"Minimum track width is {min_width_m:.2f} m, below the configured {width_threshold:.2f} m.",
                )
            )

    if enabled("min_centerline_radius"):
        radius_threshold = threshold("min_centerline_radius", 4.5)
        if geometry.min_radius_m + 1e-6 < radius_threshold:
            result.issues.append(
                make_issue(
                    "min_centerline_radius",
                    severity("min_centerline_radius"),
                    "Centerline radius below minimum",
                    f"Minimum radius is {geometry.min_radius_m:.2f} m, below the configured {radius_threshold:.2f} m.",
                )
            )

    if all_spacing.size and enabled("min_cone_spacing"):
        spacing_min_threshold = threshold("min_cone_spacing", 2.0)
        if float(np.min(all_spacing)) + 1e-6 < spacing_min_threshold:
            result.issues.append(
                make_issue(
                    "min_cone_spacing",
                    severity("min_cone_spacing"),
                    "Cone spacing below minimum",
                    f"Minimum cone spacing is {float(np.min(all_spacing)):.2f} m, below the configured {spacing_min_threshold:.2f} m.",
                )
            )

    if all_spacing.size and enabled("max_cone_spacing"):
        spacing_max_threshold = threshold("max_cone_spacing", 6.0)
        if float(np.max(all_spacing)) - 1e-6 > spacing_max_threshold:
            result.issues.append(
                make_issue(
                    "max_cone_spacing",
                    severity("max_cone_spacing"),
                    "Cone spacing above maximum",
                    f"Maximum cone spacing is {float(np.max(all_spacing)):.2f} m, above the configured {spacing_max_threshold:.2f} m.",
                )
            )

    if enabled("min_track_length"):
        track_length_threshold = threshold("min_track_length", 80.0)
        if geometry.track_length_m + 1e-6 < track_length_threshold:
            result.issues.append(
                make_issue(
                    "min_track_length",
                    severity("min_track_length"),
                    "Track length below minimum",
                    f"Track length is {geometry.track_length_m:.2f} m, below the configured {track_length_threshold:.2f} m.",
                )
            )

    return result
