from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from parameter_function import ParameterFunction
from utils_qt import (
    compute_curvature_profile,
    create_closed_spline,
    generate_variable_offset_boundaries,
    sample_cones_variable,
)

from .models import TrackGeometry


def points_to_array(points: Iterable[Tuple[float, float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(points) if not isinstance(points, np.ndarray) else points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    return array.reshape(-1, 2)


def array_to_points(points: np.ndarray) -> List[Tuple[float, float]]:
    if points is None:
        return []
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return []
    return [(float(x), float(y)) for x, y in arr.reshape(-1, 2)]


def point_centroid(points: Iterable[Tuple[float, float]] | np.ndarray) -> Tuple[float, float] | None:
    arr = points_to_array(points)
    if arr.size == 0:
        return None
    centroid = np.mean(arr, axis=0)
    return float(centroid[0]), float(centroid[1])


def transform_points(
    points: Iterable[Tuple[float, float]] | np.ndarray,
    *,
    rotation_deg: float = 0.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    translation: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
) -> np.ndarray:
    arr = points_to_array(points)
    if arr.size == 0:
        return arr

    origin_arr = np.asarray(origin, dtype=float)
    translation_arr = np.asarray(translation, dtype=float)
    transformed = (arr - origin_arr) * float(scale)

    if abs(rotation_deg) > 1e-9:
        theta = np.deg2rad(float(rotation_deg))
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=float,
        )
        transformed = transformed @ rotation.T

    return transformed + origin_arr + translation_arr


def closed_loop_segment_lengths(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.array([], dtype=float)
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if arr.shape[0] < 2:
        return np.array([], dtype=float)
    shifted = np.roll(arr, -1, axis=0)
    return np.linalg.norm(shifted - arr, axis=1)


def polyline_length_m(points: np.ndarray, px_per_m: float, closed: bool = False) -> float:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    work = arr
    if closed and not np.allclose(arr[0], arr[-1]):
        work = np.vstack((arr, arr[0]))
    diffs = np.diff(work, axis=0)
    length_px = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return length_px / float(px_per_m)


def min_radius_from_curvature(curvature_values: np.ndarray) -> float:
    values = np.asarray(curvature_values, dtype=float)
    if values.size == 0:
        return float("inf")
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("inf")
    with np.errstate(divide="ignore"):
        radii = np.where(np.abs(finite_values) > 1e-9, 1.0 / np.abs(finite_values), np.inf)
    return float(np.min(radii))


def build_track_geometry(
    control_points: List[Tuple[float, float]],
    px_per_m: float,
    n_points_midline: int,
    width_function: ParameterFunction,
    cone_spacing_function: ParameterFunction,
) -> TrackGeometry:
    geometry = TrackGeometry()
    points = points_to_array(control_points)
    if points.shape[0] < 4:
        return geometry

    centerline = create_closed_spline(points, num_points=n_points_midline)
    progress, curvature = compute_curvature_profile(centerline, px_per_m)
    left_boundary, right_boundary = generate_variable_offset_boundaries(
        centerline,
        width_function.evaluate_array,
        px_per_m,
    )

    if left_boundary is None or right_boundary is None:
        return geometry

    left_cones = sample_cones_variable(
        left_boundary,
        centerline,
        cone_spacing_function.evaluate_array,
        px_per_m,
    )
    right_cones = sample_cones_variable(
        right_boundary,
        centerline,
        cone_spacing_function.evaluate_array,
        px_per_m,
    )

    geometry.centerline = points_to_array(centerline)
    geometry.left_boundary = points_to_array(left_boundary)
    geometry.right_boundary = points_to_array(right_boundary)
    geometry.generated_left_cones = points_to_array(left_cones)
    geometry.generated_right_cones = points_to_array(right_cones)
    geometry.left_cones = points_to_array(left_cones)
    geometry.right_cones = points_to_array(right_cones)
    geometry.curvature_progress = np.asarray(progress, dtype=float)
    geometry.curvature_values = np.asarray(curvature, dtype=float)
    geometry.track_length_m = polyline_length_m(geometry.centerline, px_per_m, closed=True)
    geometry.min_radius_m = min_radius_from_curvature(geometry.curvature_values)
    return geometry


def set_cone_overrides(
    geometry: TrackGeometry,
    left_cones: List[Tuple[float, float]] | np.ndarray,
    right_cones: List[Tuple[float, float]] | np.ndarray,
) -> None:
    left = points_to_array(left_cones)
    right = points_to_array(right_cones)
    geometry.left_cones = left if left.size else np.empty((0, 2), dtype=float)
    geometry.right_cones = right if right.size else np.empty((0, 2), dtype=float)
