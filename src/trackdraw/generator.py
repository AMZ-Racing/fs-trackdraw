from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

from parameter_function import ParameterFunction

from .geometry import build_track_geometry
from .models import GeneratorSettings, RuleSettings
from .validation import validate_track


@dataclass
class GeneratorResult:
    control_points: List[Tuple[float, float]]
    succeeded: bool
    message: str


COMPLEXITY_TO_POINTS = {
    "simple": 7,
    "balanced": 9,
    "complex": 12,
}


GENERATOR_SAMPLE_POINTS = 180


def _extract_polygon(feature) -> Polygon:
    if feature.geom_type == "Polygon":
        return feature
    polygons = [geom for geom in getattr(feature, "geoms", []) if geom.geom_type == "Polygon"]
    if not polygons:
        return Polygon()
    polygons.sort(key=lambda geom: geom.area, reverse=True)
    return polygons[0]


def generate_control_points(
    allowed_area_points: List[Tuple[float, float]],
    px_per_m: float,
    width_function: ParameterFunction,
    cone_spacing_function: ParameterFunction,
    n_points_midline: int,
    rules: Dict[str, RuleSettings],
    settings: GeneratorSettings,
) -> GeneratorResult:
    area = Polygon(allowed_area_points)
    if not area.is_valid or area.area <= 0:
        return GeneratorResult([], False, "Allowed area is invalid.")

    track_width = max(0.1, width_function.evaluate(0.0))
    inward_margin_px = max((track_width * 0.5 + settings.min_clearance_m) * px_per_m, 12.0)
    feasible = _extract_polygon(area.buffer(-inward_margin_px, join_style=2))
    if feasible.is_empty or feasible.area <= 0:
        feasible = _extract_polygon(area.buffer(-0.35 * track_width * px_per_m, join_style=2))
    if feasible.is_empty or feasible.area <= 0:
        return GeneratorResult([], False, "Allowed area is too small for the current width and clearance.")

    boundary = feasible.exterior
    perimeter = boundary.length
    anchor = np.array(feasible.representative_point().coords[0], dtype=float)
    complexity_key = settings.complexity if settings.complexity in COMPLEXITY_TO_POINTS else "balanced"
    n_control_points = COMPLEXITY_TO_POINTS[complexity_key]
    rng = np.random.default_rng(int(settings.seed))
    sample_points = max(96, min(int(n_points_midline), GENERATOR_SAMPLE_POINTS))
    deadline = time.monotonic() + max(0.05, float(settings.timeout_s))

    best_points: List[Tuple[float, float]] = []
    best_score = None
    best_message = "Generator could not find a valid track."

    for attempt in range(max(1, int(settings.attempts))):
        if attempt and time.monotonic() >= deadline:
            timeout_message = "Generator timed out before finding a fully valid track."
            if best_points:
                return GeneratorResult(best_points, False, timeout_message)
            return GeneratorResult([], False, timeout_message)

        phase = rng.uniform(0.0, perimeter)
        factors = rng.uniform(0.42, 0.78, size=n_control_points)
        tangential_jitter = rng.uniform(-0.04, 0.04, size=n_control_points)
        distances = np.linspace(0.0, perimeter, n_control_points, endpoint=False)
        distances = (distances + phase + tangential_jitter * perimeter) % perimeter

        control_points: List[Tuple[float, float]] = []
        for distance, factor in zip(distances, factors):
            boundary_point = np.array(boundary.interpolate(float(distance)).coords[0], dtype=float)
            for shrink in np.linspace(factor, 0.2, 10):
                candidate = anchor + (boundary_point - anchor) * shrink
                if feasible.covers(Point(float(candidate[0]), float(candidate[1]))):
                    control_points.append((float(candidate[0]), float(candidate[1])))
                    break

        if len(control_points) < 4:
            continue

        geometry = build_track_geometry(
            control_points=control_points,
            px_per_m=px_per_m,
            n_points_midline=sample_points,
            width_function=width_function,
            cone_spacing_function=cone_spacing_function,
        )
        validation = validate_track(allowed_area_points, geometry, px_per_m, rules)
        score = validation.error_count * 100 + validation.warning_count

        if best_score is None or score < best_score:
            best_score = score
            best_points = control_points
            if validation.issues:
                best_message = validation.issues[0].detail
            else:
                best_message = "Generated a valid track."

        if validation.is_valid:
            return GeneratorResult(control_points, True, "Generated a valid track.")

    return GeneratorResult(best_points, False, best_message)
