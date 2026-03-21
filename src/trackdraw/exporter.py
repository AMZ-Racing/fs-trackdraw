from __future__ import annotations

import csv
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def transform_points_to_export_frame(
    centerline: Sequence[Sequence[float]],
    points: Iterable[Sequence[float]],
    px_per_m: float,
) -> List[Tuple[float, float]]:
    center = np.asarray(centerline, dtype=float)
    if center.ndim != 2 or center.shape[0] < 2:
        raise ValueError("Not enough centerline points to define export frame")

    origin = center[0]
    tangent = center[1] - origin
    theta = math.atan2(float(tangent[1]), float(tangent[0]))
    rotation = np.array(
        [
            [math.cos(-theta), -math.sin(-theta)],
            [math.sin(-theta), math.cos(-theta)],
        ],
        dtype=float,
    )

    transformed: List[Tuple[float, float]] = []
    for point in points:
        local = np.asarray(point, dtype=float) - origin
        local_rot = rotation.dot(local) / float(px_per_m)
        transformed.append((float(local_rot[0]), float(local_rot[1])))
    return transformed


def export_track_csv(
    filename: str,
    centerline: Sequence[Sequence[float]],
    left_cones: Sequence[Sequence[float]],
    right_cones: Sequence[Sequence[float]],
    px_per_m: float,
) -> None:
    left_cones_m = transform_points_to_export_frame(centerline, left_cones, px_per_m)
    right_cones_m = transform_points_to_export_frame(centerline, right_cones, px_per_m)

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tag", "x", "y"])
        for point in right_cones_m:
            writer.writerow(["yellow", point[0], point[1]])
        for point in left_cones_m:
            writer.writerow(["blue", point[0], point[1]])
