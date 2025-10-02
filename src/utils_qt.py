import math
from typing import Callable, List, Tuple

import numpy as np
from shapely.geometry import LineString
from scipy.interpolate import splprep, splev


def create_closed_spline(control_points, num_points=100):
    """
    Compute a smooth, closed B-spline from a list of control points.
    """
    pts = np.array(control_points)
    # add first point to end to close the loop
    pts = np.vstack((pts, pts[0]))
    # if we have only one point, return it
    if len(pts) <= 2:
        return pts
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine = splev(u_fine, tck)
    return np.column_stack((x_fine, y_fine))

def robust_parallel_offset(ls, distance, side, join_style=2):
    """
    Compute a parallel offset of a LineString using shapely's parallel_offset.
    If the result is a MultiLineString, return the longest LineString.
    """
    offset = ls.parallel_offset(distance, side, join_style=join_style)
    if offset.is_empty:
        return None
    if offset.geom_type == "MultiLineString":
      lines = list(offset.geoms)  # Use .geoms to access individual LineStrings.
      lines.sort(key=lambda l: l.length, reverse=True)
      return lines[0]
    return offset

def generate_offset_boundaries(track_points, track_width_meters, px_per_m):
    """
    Compute left and right boundaries as parallel offsets from the centerline.
    """
    track_width_px = track_width_meters * px_per_m
    ls = LineString(track_points)
    left_offset = robust_parallel_offset(ls, track_width_px / 2.0, 'left', join_style=2)
    right_offset = robust_parallel_offset(ls, track_width_px / 2.0, 'right', join_style=2)
    if left_offset is None or right_offset is None:
        return None, None
    left_coords = list(left_offset.coords)
    right_coords = list(right_offset.coords)
    return np.array(left_coords), np.array(right_coords)

def generate_oneside_boundary(points, offset_meters, px_per_m):
    """
    Generate a one-sided boundary by offsetting the given points.
    """
    offset_px = offset_meters * px_per_m
    ls = LineString(points)
    left_offset = robust_parallel_offset(ls, offset_px, 'left', join_style=2)
    if left_offset is None:
        return None
    left_coords = list(left_offset.coords)
    return np.array(left_coords)

def sample_cones(boundary, cone_spacing_meters, px_per_m):
    """Sample points along a boundary so that they are approximately cone_spacing_meters apart."""
    cone_spacing_px = cone_spacing_meters * px_per_m
    pts = boundary
    if len(pts) < 2:
        return pts
    distances = [0]
    for i in range(1, len(pts)):
        d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        distances.append(distances[-1] + d)
    total_length = distances[-1]
    num_cones = max(2, int(total_length // cone_spacing_px))
    sample_d = np.linspace(0, total_length, num_cones)
    sampled = []
    for sd in sample_d:
        for i in range(1, len(distances)):
            if distances[i] >= sd:
                t = (sd - distances[i-1]) / (distances[i] - distances[i-1])
                x = pts[i-1][0] + t * (pts[i][0] - pts[i-1][0])
                y = pts[i-1][1] + t * (pts[i][1] - pts[i-1][1])
                sampled.append((x, y))
                break

    # Now, remove the last cone if it's too close to the first one
    if len(sampled) > 1 and np.linalg.norm(np.array(sampled[0]) - np.array(sampled[-1])) < cone_spacing_meters:
        sampled = sampled[:-1]

    return np.array(sampled)


def _remove_duplicate_endpoint(points: np.ndarray) -> np.ndarray:
    """Remove duplicate final point for closed loops (if present)."""
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        return points[:-1]
    return points


def _closed_path_metrics(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return cumulative distances (per vertex), segment lengths, and total length for a closed path."""
    n = len(points)
    if n < 2:
        return np.zeros(1), np.zeros(1), 0.0
    indices = np.arange(n)
    next_indices = (indices + 1) % n
    segment_vecs = points[next_indices] - points[indices]
    segment_lengths = np.linalg.norm(segment_vecs, axis=1)
    cumulative = np.zeros(n + 1)
    cumulative[1:] = np.cumsum(segment_lengths)
    total = cumulative[-1]
    return cumulative, segment_lengths, total


def _interpolate_on_closed_path(points: np.ndarray, cumulative: np.ndarray, segment_lengths: np.ndarray,
                                target_distance: float) -> np.ndarray:
    """Interpolate a point located at target_distance along the closed path."""
    total_length = cumulative[-1]
    if total_length == 0:
        return points[0]
    # Wrap target distance into path length
    t = target_distance % total_length
    idx = np.searchsorted(cumulative, t, side='right') - 1
    idx = int(np.clip(idx, 0, len(points) - 1))
    seg_len = segment_lengths[idx]
    if seg_len == 0:
        return points[idx]
    local_dist = t - cumulative[idx]
    frac = local_dist / seg_len
    next_idx = (idx + 1) % len(points)
    return points[idx] + frac * (points[next_idx] - points[idx])


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Return the scalar 2D cross product of vectors a and b."""
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection(p1: np.ndarray,
                          p2: np.ndarray,
                          p3: np.ndarray,
                          p4: np.ndarray,
                          tol: float = 1e-9):
    """Return (point, t, u) if segments p1-p2 and p3-p4 intersect, else None."""
    r = p2 - p1
    s = p4 - p3
    denom = _cross_2d(r, s)
    if abs(denom) < tol:
        return None

    diff = p3 - p1
    t = _cross_2d(diff, s) / denom
    u = _cross_2d(diff, r) / denom

    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        # Clamp within segment bounds to guard against floating error
        t_clamped = min(max(t, 0.0), 1.0)
        u_clamped = min(max(u, 0.0), 1.0)
        point = p1 + t_clamped * r
        return point, t_clamped, u_clamped
    return None


def _find_self_intersections(points: np.ndarray,
                             cumulative: np.ndarray,
                             segment_lengths: np.ndarray,
                             total_length: float,
                             dedup_tol: float = 1.0,
                             dedup_progress_tol: float = 1.0,
                             tol: float = 1e-6) -> List[dict]:
    """Return list of {'point': np.ndarray, 'distance': float} for boundary self-intersections."""
    n = len(points)
    if n < 4 or total_length <= 0:
        return []

    intersections: List[dict] = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i:
                continue
            if j == (i + 1) % n or i == (j + 1) % n:
                continue  # adjacent segments share endpoints

            q1 = points[j]
            q2 = points[(j + 1) % n]

            result = _segment_intersection(p1, p2, q1, q2, tol=tol)
            if result is None:
                continue

            point, t_param, u_param = result
            if segment_lengths[i] <= tol:
                continue

            distance = (cumulative[i] + t_param * segment_lengths[i]) % total_length
            if total_length - distance < tol:
                distance = 0.0

            # Deduplicate intersections that land almost at existing vertices
            if np.any(np.linalg.norm(points - point, axis=1) < tol):
                continue

            def _already_present(dist_val: float) -> bool:
                for existing in intersections:
                    if np.linalg.norm(existing["point"] - point) < dedup_tol:
                        dist_diff = abs(existing["distance"] - dist_val)
                        wrap_diff = min(dist_diff, total_length - dist_diff)
                        if wrap_diff < dedup_progress_tol:
                            return True
                return False

            if not _already_present(distance):
                intersections.append({"point": point, "distance": distance})

            if segment_lengths[j] <= tol:
                continue

            distance_j = (cumulative[j] + u_param * segment_lengths[j]) % total_length
            if total_length - distance_j < tol:
                distance_j = 0.0

            if not _already_present(distance_j):
                intersections.append({"point": point, "distance": distance_j})

    if not intersections:
        return []

    intersections.sort(key=lambda item: item["distance"])
    return intersections


def _filter_cones_by_knot_regions(cone_positions: np.ndarray,
                                  cone_distances: np.ndarray,
                                  knots: List[dict],
                                  eps: float = 1e-6) -> np.ndarray:
    """Toggle cone placement whenever the boundary crosses itself."""
    if cone_positions.size == 0 or not knots:
        return cone_positions

    events: List[Tuple[float, int, np.ndarray]] = []
    for knot in knots:
        events.append((float(knot["distance"]), 0, np.asarray(knot["point"], dtype=float)))
    for distance, point in zip(cone_distances, cone_positions):
        events.append((float(distance), 1, np.asarray(point, dtype=float)))

    events.sort(key=lambda item: (item[0], item[1]))

    result: List[np.ndarray] = []
    valid = True

    def _append_unique(pt: np.ndarray):
        if not result:
            result.append(pt)
            return
        if np.linalg.norm(result[-1] - pt) > eps:
            result.append(pt)

    for _, event_type, point in events:
        if event_type == 0:  # knot
            _append_unique(point)
            valid = not valid
        else:  # cone
            if valid:
                _append_unique(point)

    if not result:
        return cone_positions

    return np.asarray(result, dtype=float)


def generate_variable_offset_boundaries(centerline: np.ndarray,
                                        width_sampler: Callable[[np.ndarray], np.ndarray],
                                        px_per_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute left/right boundaries for a centerline with variable track width."""
    if centerline is None or len(centerline) < 2:
        return None, None

    base_points = _remove_duplicate_endpoint(np.asarray(centerline, dtype=float))
    if len(base_points) < 3:
        return None, None

    cumulative, _, total_length = _closed_path_metrics(base_points)
    if total_length == 0:
        return None, None

    progress = cumulative[:-1] / total_length
    widths_m = width_sampler(progress)
    widths_px = widths_m * px_per_m

    left_pts = []
    right_pts = []
    n = len(base_points)
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        prev_pt = base_points[prev_idx]
        next_pt = base_points[next_idx]
        tangent = next_pt - prev_pt
        norm = np.linalg.norm(tangent)
        if norm == 0:
            # fallback to difference with next point only
            tangent = base_points[next_idx] - base_points[i]
            norm = np.linalg.norm(tangent)
            if norm == 0:
                tangent = np.array([1.0, 0.0])
                norm = 1.0
        tangent_unit = tangent / norm
        normal = np.array([-tangent_unit[1], tangent_unit[0]])
        half_width = widths_px[i] / 2.0
        left_pts.append(base_points[i] + normal * half_width)
        right_pts.append(base_points[i] - normal * half_width)

    left_pts = np.asarray(left_pts)
    right_pts = np.asarray(right_pts)
    # Close the loops by appending the first point
    left_pts = np.vstack((left_pts, left_pts[0]))
    right_pts = np.vstack((right_pts, right_pts[0]))
    return left_pts, right_pts


def sample_cones_variable(boundary: np.ndarray,
                          centerline: np.ndarray,
                          spacing_sampler: Callable[[np.ndarray], np.ndarray],
                          px_per_m: float,
                          min_spacing_m: float = 0.5) -> np.ndarray:
    """Sample cone positions using a spacing function defined over track progress."""
    if boundary is None or centerline is None or len(centerline) < 2:
        return np.array([])

    boundary_points = _remove_duplicate_endpoint(np.asarray(boundary, dtype=float))
    centerline_points = _remove_duplicate_endpoint(np.asarray(centerline, dtype=float))
    if len(centerline_points) < 3 or len(boundary_points) != len(centerline_points):
        return np.array([])

    cumulative, segment_lengths, total_length = _closed_path_metrics(centerline_points)
    if total_length == 0:
        return np.array([])

    min_spacing_px = max(min_spacing_m * px_per_m, 1.0)

    cone_positions = []
    cone_distances = []
    # Always include start cone
    start_point = boundary_points[0]
    cone_positions.append(start_point)
    cone_distances.append(0.0)

    next_distance = spacing_sampler(np.array([0.0], dtype=float))[0] * px_per_m
    if not np.isfinite(next_distance) or next_distance <= 0:
        next_distance = min_spacing_px
    else:
        next_distance = max(next_distance, min_spacing_px)

    while next_distance < total_length:
        cone_point = _interpolate_on_closed_path(boundary_points, cumulative, segment_lengths, next_distance)
        cone_positions.append(cone_point)
        cone_distances.append(next_distance % total_length)
        progress = next_distance / total_length
        spacing_px = spacing_sampler(np.array([progress], dtype=float))[0] * px_per_m
        if not np.isfinite(spacing_px) or spacing_px <= 0:
            spacing_px = min_spacing_px
        else:
            spacing_px = max(spacing_px, min_spacing_px)
        next_distance += spacing_px

    # Drop last cone if it's too close to the first
    if len(cone_positions) > 1:
        distance_to_first = np.linalg.norm(cone_positions[-1] - cone_positions[0])
        if distance_to_first < min_spacing_m * px_per_m * 0.5:
            cone_positions.pop()
            cone_distances.pop()

    cone_positions_arr = np.asarray(cone_positions, dtype=float)
    cone_distances_arr = np.asarray(cone_distances, dtype=float)

    knots = _find_self_intersections(boundary_points, cumulative, segment_lengths, total_length)
    if knots:
        cone_positions_arr = _filter_cones_by_knot_regions(
            cone_positions_arr,
            cone_distances_arr,
            knots,
        )

    return cone_positions_arr


def compute_curvature_profile(centerline: np.ndarray, px_per_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (progress, curvature) with progress in [0,1) and curvature in 1/m."""
    if centerline is None:
        return np.array([]), np.array([])

    pts = _remove_duplicate_endpoint(np.asarray(centerline, dtype=float))
    if len(pts) < 3:
        return np.array([]), np.array([])

    cumulative, _, total_length = _closed_path_metrics(pts)
    if total_length <= 0:
        return np.array([]), np.array([])

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    except ValueError:
        return np.array([]), np.array([])

    sample_count = len(pts)
    u_samples = np.linspace(0, 1, sample_count, endpoint=False)
    dx, dy = splev(u_samples, tck, der=1)
    ddx, ddy = splev(u_samples, tck, der=2)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature_px = np.where(denominator > 1e-12, numerator / denominator, 0.0)

    curvature_m = curvature_px * px_per_m
    progress = cumulative[:-1] / total_length
    return progress.astype(float), curvature_m.astype(float)
