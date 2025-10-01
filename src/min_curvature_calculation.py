"""
Raceline Optimizer

This implements an optimization routine to compute the optimal racing
line between two bounding walls (left and right) of a track that can be
closed or open

Algorithm Overview:
1. Build an initial centreline by pairing points on left and right walls.
2. Iteratively refine the racing line:
   a. Resample the current reference curve at uniform arc‑length spacing.
   b. Compute tangent and normal vectors via finite differences
   c. Measure left/right clearances by ray‑casting to walls
      (KD‑tree or brute force).
   d. Formulate and solve a quadratic program (QP) that trades off:
      - matching the track's curvature   (K · offsets ≈ reference curvature)
      - magnitude of offsets (regularization)
   e. Apply lateral offsets along normals to update reference curve.
3. After all iterations, resample the final curve and return:
   - x, y coordinates of the optimized raceline
   - left/right clearances (w_left, w_right)
   - curvature at each sample point
   - cumulative arc‑length s along the raceline

Usage Example:
    from raceline_optimizer import optimize_raceline

    # walls: np.ndarray of shape (N,2) and (M,2)
    x_opt, y_opt, w_left, w_right, curvature, s = optimize_raceline(
        wall_left, wall_right,
        spacing_list=[1.5, 1.0, 0.5],
        num_iterations=3, # If left as None, will be 3 automatically
        dist_weight=1e-5,
        wall_clearance=0.5,
        use_sparse=True,
        use_kdtree=True,
        closed_path=True,
        debug=True
    )

    The dist_weight makes the optimization less aggressive, so the new
    solution after each iteration is closer to the previous one. This is
    useful if the optimization becomes unstable.  Another tool for
    unstable optimization is the spacing used. The smaller the spacing,
    the more unstable. That is why it is only good to use small spacing
    at the last iterations. That also saves time of course.
"""

import time
from typing import Union, Sequence, Tuple
import numpy as np
import cvxpy as cp
from scipy.spatial import cKDTree
from scipy.sparse import diags


def resample_curve(
    x: np.ndarray, y: np.ndarray, ds: float, closed: bool
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Uniformly resamples a line (x, y) at spacing ds.

    If *closed* is True the curve is treated as periodic by appending the
    first point to the end before computing arc‑length, ensuring continuity.
    For an open curve the endpoints are not joined.
    """
    if closed:
        x_work = np.concatenate([x, x[:1]])
        y_work = np.concatenate([y, y[:1]])
    else:
        x_work = x.copy()
        y_work = y.copy()

    dx = np.diff(x_work)
    dy = np.diff(y_work)
    seg_len = np.hypot(dx, dy)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = s_cum[-1]

    # Number of segments of length ~ds (keep last original point for open)
    num_points = int(np.floor(total_len / ds))
    if closed:
        s_uniform = np.linspace(0.0, total_len, num_points + 1, endpoint=False)
    else:
        s_uniform = np.linspace(0.0, total_len, num_points + 1, endpoint=True)

    x_uni = np.interp(s_uniform, s_cum, x_work)
    y_uni = np.interp(s_uniform, s_cum, y_work)

    return x_uni, y_uni, total_len


def tangents_and_normals(
    x: np.ndarray, y: np.ndarray, closed: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes unit tangent and normal vectors at each point of a curve.

    Uses central differences
    """
    if closed:
        dx = np.roll(x, -1) - np.roll(x, +1)
        dy = np.roll(y, -1) - np.roll(y, +1)
        dx *= 0.5
        dy *= 0.5
    else:
        dx = np.gradient(x)
        dy = np.gradient(y)

    speed = np.hypot(dx, dy) + 1e-12
    tx = dx / speed
    ty = dy / speed

    nx = -ty  # outward normal by rotating tangent +90°
    ny = tx

    tangents = np.column_stack((tx, ty))
    normals = np.column_stack((nx, ny))
    return tangents, normals


def curvature_of_polyline(x: np.ndarray, y: np.ndarray, ds: float, closed: bool) -> np.ndarray:
    """
    Computes curvature κ of a polyline.

    κ = (x' y'' − y' x'') / ((x'² + y'²)^(3/2)).
    """
    if closed:
        dx = (np.roll(x, -1) - np.roll(x, +1)) / (2 * ds)
        dy = (np.roll(y, -1) - np.roll(y, +1)) / (2 * ds)
        ddx = (np.roll(dx, -1) - np.roll(dx, +1)) / (2 * ds)
        ddy = (np.roll(dy, -1) - np.roll(dy, +1)) / (2 * ds)
    else:
        dx = np.gradient(x, ds)
        dy = np.gradient(y, ds)
        ddx = np.gradient(dx, ds)
        ddy = np.gradient(dy, ds)

    numerator = dx * ddy - dy * ddx
    denominator = (dx * dx + dy * dy) ** 1.5 + 1e-12
    return numerator / denominator


def build_second_diff_matrix(n: int, ds: float, closed: bool, sparse: bool = True):
    """
    Builds the second‑difference matrix **K** (n×n).

    closed selects periodic (wrap‑around) or open (clamped) structure.
    """
    if closed:
        if sparse:
            e = np.ones(n)
            offsets = [0, 1, -1]
            data = [2 * e, -e, -e]
            K = diags(data, offsets, shape=(n, n), format="lil")
            K[0, n - 1] = -1.0
            K[n - 1, 0] = -1.0
            return K.tocsc() / (ds * ds)
        else:
            K = np.zeros((n, n))
            for i in range(n):
                im1 = (i - 1) % n
                ip1 = (i + 1) % n
                K[i, i] = 2.0
                K[i, im1] = -1.0
                K[i, ip1] = -1.0
            return K / (ds * ds)

    else:  # open curve
        if sparse:
            e = np.ones(n)
            offsets = [0, 1, -1]
            data = [-2 * e, e, e]
            K = diags(data, offsets, shape=(n, n), format="lil")
            K[0, 0] = -1.0
            K[0, 1] = 1.0
            K[n - 1, n - 1] = -1.0
            K[n - 1, n - 2] = 1.0
            return K.tocsc() / (ds * ds)
        else:
            K = np.zeros((n, n))
            for i in range(1, n - 1):
                K[i, i] = -2.0
                K[i, i - 1] = 1.0
                K[i, i + 1] = 1.0
            # forward/backward difference for endpoints
            K[0, 0] = -1.0
            K[0, 1] = 1.0
            K[n - 1, n - 1] = -1.0
            K[n - 1, n - 2] = 1.0
            return K / (ds * ds)


def widths_by_ray_cast(
    path: np.ndarray,
    normals: np.ndarray,
    wall_left: np.ndarray,
    wall_right: np.ndarray,
    use_kdtree: bool = True,
    search_radius: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes left/right clearance width at each point by ray‑casting.
    """
    n = len(path)
    wL = np.zeros(n)
    wR = np.zeros(n)
    eps = 1e-6
    tol = 1e-6

    if use_kdtree:
        centL = 0.5 * (wall_left[:-1] + wall_left[1:])
        centR = 0.5 * (wall_right[:-1] + wall_right[1:])
        treeL = cKDTree(centL)
        treeR = cKDTree(centR)

        if search_radius <= 0.0:
            dists, _ = treeL.query(path)
            search_radius = np.max(dists) * 1.005

    for i, (P, nh) in enumerate(zip(path, normals)):
        # left clearance
        λ_min = np.inf
        P0 = P + nh * eps
        if use_kdtree:
            idxs = treeL.query_ball_point(P, r=search_radius)
            segments = idxs
        else:
            segments = range(len(wall_left) - 1)
        for idx in segments:
            A, B = wall_left[idx], wall_left[idx + 1]
            M = np.column_stack((nh, A - B))
            if abs(np.linalg.det(M)) < tol:
                continue
            λ, t = np.linalg.solve(M, A - P0)
            if λ >= -tol and 0 <= t <= 1 + tol:
                λ_min = min(λ_min, max(0.0, λ))
        wL[i] = 0.0 if not np.isfinite(λ_min) else λ_min

        # right clearance
        λ_min = np.inf
        P0 = P - nh * eps
        if use_kdtree:
            idxs = treeR.query_ball_point(P, r=search_radius)
            segments = idxs
        else:
            segments = range(len(wall_right) - 1)
        for idx in segments:
            A, B = wall_right[idx], wall_right[idx + 1]
            M = np.column_stack((-nh, A - B))
            if abs(np.linalg.det(M)) < tol:
                continue
            λ, t = np.linalg.solve(M, A - P0)
            if λ >= -tol and 0 <= t <= 1 + tol:
                λ_min = min(λ_min, max(0.0, λ))
        wR[i] = 0.0 if not np.isfinite(λ_min) else λ_min

    return wL, wR


def compute_initial_centreline(
    wall_left: np.ndarray, wall_right: np.ndarray, ds: float, closed: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the initial centreline by pairing nearest points
    on the left and right walls and resampling at spacing ``ds``.
    """
    # Resample walls (open/closed handled inside)
    lx, ly, _ = resample_curve(wall_left[:, 0], wall_left[:, 1], ds, closed)
    rx, ry, _ = resample_curve(wall_right[:, 0], wall_right[:, 1], ds, closed)
    ptsL = np.column_stack((lx, ly))
    ptsR = np.column_stack((rx, ry))
    treeR = cKDTree(ptsR)
    _, idxR = treeR.query(ptsL)
    raw_ctr = 0.5 * (ptsL + ptsR[idxR])
    cx, cy, _ = resample_curve(raw_ctr[:, 0], raw_ctr[:, 1], ds, closed)
    return cx, cy


# ---------------------------------------------------------------------------
# Main optimisation routine
# ---------------------------------------------------------------------------


def optimize_raceline(
    wall_left: np.ndarray,
    wall_right: np.ndarray,
    spacing_list: Union[float, Sequence[float]] = 1.0,
    num_iterations: int = 3,
    dist_weight: float = 1e-5,
    wall_clearance: float = 0.1,
    use_sparse: bool = True,
    use_kdtree: bool = True,
    closed_path: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the optimal racing line between two walls.

    Parameters:
        wall_left, wall_right : Arrays (N,2), (M, 2) defining the track boundaries.
        spacing_list          : Single float or list of floats for each iter.
        num_iterations        : Number of refinement iterations.
        dist_weight           : Regularization weight on lateral offsets.
        wall_clearance        : Minimum clearance maintained to walls.
        use_sparse            : Use sparse matrix for K if True.
        use_kdtree            : Use KD-tree for ray-casting if True.
        closed_path           : Treat the path as closed (periodic) if True.
        debug                 : Print debug information if True.

    Returns:
        x_opt, y_opt       : Coordinates of optimized raceline.
        w_left, w_right    : Left/right clearance distances.
        curvature          : Curvature at each sample.
        s                  : Cumulative arc‑length along the raceline.
    """
    if debug:
        t_total = time.time()

    # ---- build list of spacings --------------------------------------------
    if isinstance(spacing_list, (float, int)):
        ds_list = [float(spacing_list)] * num_iterations
    else:
        ds_list = list(spacing_list)
        if num_iterations is None:
            num_iterations = len(ds_list)
        if len(ds_list) != num_iterations:
            raise ValueError("Length of spacing_list must equal num_iterations")

    # ---- initial centreline -------------------------------------------------
    if debug:
        t = time.time()
    if closed_path:
        wall_left = np.vstack([wall_left, wall_left[0]])
        wall_right = np.vstack([wall_right, wall_right[0]])
    ds0 = ds_list[0]
    x_ref, y_ref = compute_initial_centreline(wall_left, wall_right, ds0, closed_path)
    if debug:
        print(f"Initial centreline computed in {time.time() - t:.3f} s")

    alpha = None
    normals = None

    # ========================================================================
    #                         refinement iterations
    # ========================================================================
    for i in range(num_iterations):
        ds = ds_list[i]
        if debug:
            t_start = time.time()
            print(f"Iteration {i + 1}/{num_iterations}, ds={ds}", end="")

        # ---- apply previous offsets ----------------------------------------
        if alpha is not None:
            x_ref = x_ref + normals[:, 0] * alpha
            y_ref = y_ref + normals[:, 1] * alpha

        # ---- resample reference line ---------------------------------------
        if debug:
            t = time.time()
        x_ref, y_ref, _ = resample_curve(x_ref, y_ref, ds, closed_path)
        n = len(x_ref)
        if debug:
            print(f" point count: {n}")
            print(f"  Resampling done in {time.time() - t:.3f} s")

        # ---- compute tangents & normals ------------------------------------
        tang, normals = tangents_and_normals(x_ref, y_ref, closed_path)

        # ---- left/right clearances -----------------------------------------
        if debug:
            t = time.time()
        path = np.column_stack((x_ref, y_ref))
        wL, wR = widths_by_ray_cast(path, normals, wall_left, wall_right, use_kdtree)
        if debug:
            print(f"  Ray‑casting done in {time.time() - t:.3f} s")

        # ---- build curvature operator K ------------------------------------
        if debug:
            t = time.time()
        K = build_second_diff_matrix(n, ds, closed_path, use_sparse)
        if debug:
            print(f"  K matrix built in {time.time() - t:.3f} s")

        # ---- reference curvature -------------------------------------------
        curv = curvature_of_polyline(x_ref, y_ref, ds, closed_path)

        # ---- solve quadratic program ---------------------------------------
        if debug:
            t = time.time()
        alpha_var = cp.Variable(n)
        cost = cp.sum_squares(curv - K @ alpha_var) + dist_weight * cp.sum_squares(alpha_var)
        constraints = [
            alpha_var >= -np.maximum(0.0, wR - wall_clearance),
            alpha_var <= np.maximum(0.0, wL - wall_clearance),
        ]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver="OSQP", warm_start=True, verbose=False)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError("QP did not solve optimally")

        alpha = alpha_var.value

        if debug:
            print(f"  QP solved in {time.time() - t:.3f} s")
            print(f"    Iteration done in {time.time() - t_start:.3f} s")

    if debug:
        t = time.time()
    x_opt, y_opt, _ = resample_curve(
        x_ref + normals[:, 0] * alpha, y_ref + normals[:, 1] * alpha, ds_list[-1], closed_path
    )
    tang_opt, normals_opt = tangents_and_normals(x_opt, y_opt, closed_path)
    w_left, w_right = widths_by_ray_cast(
        np.column_stack((x_opt, y_opt)), normals_opt, wall_left, wall_right, use_kdtree
    )
    curvature = curvature_of_polyline(x_opt, y_opt, ds_list[-1], closed_path)

    # ---- cumulative distance ----------------------------------------------
    if closed_path:
        dx = np.diff(x_opt, append=x_opt[0])
        dy = np.diff(y_opt, append=y_opt[0])
    else:
        dx = np.diff(x_opt)
        dy = np.diff(y_opt)
    dist_seg = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(dist_seg)])[:-1]

    if debug:
        print(f"Final raceline recomputed in {time.time() - t:.3f} s")
        print(f"Total time: {time.time() - t_total:.3f} s")

    return x_opt, y_opt, w_left, w_right, curvature, s
    



if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    csv_path = "trackfiles_csv/ipz_example.csv"
    df = pd.read_csv(csv_path)
    grouped = df.groupby("tag")[["x", "y"]]
    walls = {tag: group.values for tag, group in grouped}
    wall_left = walls["blue"]
    wall_right = walls["yellow"]

    x_opt, y_opt, wL, wR, curv, s = optimize_raceline(
        wall_left,
        wall_right,
        spacing_list=[1.5, 1.0, 0.5],  # Number of iterations will be 3
        dist_weight=1e-5,
        wall_clearance=1.0,
        use_sparse=True,
        use_kdtree=True,
        closed_path=True,
        debug=True,
    )

    # Normals for plotting
    tang_opt, normals_opt = tangents_and_normals(x_opt, y_opt, closed=True)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(wall_left[:, 0], wall_left[:, 1], "gray", lw=0.7, label="Left wall")
    plt.plot(wall_right[:, 0], wall_right[:, 1], "gray", lw=0.7, label="Right wall")
    plt.plot(x_opt, y_opt, "r-", lw=2, label="Optimized Raceline")
    plt.scatter(
        x_opt + normals_opt[:, 0] * wL,
        y_opt + normals_opt[:, 1] * wL,
        c="blue",
        s=4,
        label="Left Clearance",
    )
    plt.scatter(
        x_opt - normals_opt[:, 0] * wR,
        y_opt - normals_opt[:, 1] * wR,
        c="green",
        s=4,
        label="Right Clearance",
    )
    plt.axis("equal")
    plt.legend()
    plt.title("Optimized Raceline")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.tight_layout()
    plt.show()
