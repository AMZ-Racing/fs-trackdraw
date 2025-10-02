import numpy as np


class ParameterFunction:
    """Represent a parameter that varies as a function of normalized track progress."""

    def __init__(self, default_value: float, name: str = "parameter"):
        self.name = name
        self._default_value = float(default_value)
        self._control_points = [(0.0, self._default_value), (1.0, self._default_value)]
        self._xp = None
        self._yp = None
        self._rebuild()

    @property
    def default_value(self) -> float:
        return self._default_value

    def set_constant(self, value: float) -> None:
        """Collapse to a constant function."""
        val = float(value)
        self._control_points = [(0.0, val), (1.0, val)]
        self._default_value = val
        self._rebuild()

    def set_control_points(self, control_points) -> None:
        """Replace control points with a new collection of (progress, value)."""
        if not control_points:
            # fallback to default constant if nothing provided
            self.set_constant(self._default_value)
            return
        cleaned = []
        for progress, value in control_points:
            try:
                p = float(progress)
                v = float(value)
            except (TypeError, ValueError):
                continue
            # wrap progress into [0, 1)
            p_wrapped = p % 1.0
            cleaned.append((p_wrapped, v))
        if not cleaned:
            self.set_constant(self._default_value)
            return
        # ensure sorted by progress
        cleaned.sort(key=lambda item: item[0])
        # merge duplicates by averaging values
        merged = []
        for p, v in cleaned:
            if not merged or abs(p - merged[-1][0]) > 1e-6:
                merged.append([p, v])
            else:
                merged[-1][1] = 0.5 * (merged[-1][1] + v)
        # guarantee endpoints at 0 and 1
        if merged[0][0] > 1e-6:
            merged.insert(0, [0.0, merged[0][1]])
        if merged[-1][0] < 1.0 - 1e-6:
            merged.append([1.0, merged[-1][1]])
        else:
            # snap to 1 exactly for stability
            merged[-1][0] = 1.0
        # enforce equality of first and last values for periodic continuity
        merged[-1][1] = merged[0][1]
        self._control_points = [(p, v) for p, v in merged]
        self._rebuild()

    def get_control_points(self):
        return [(float(p), float(v)) for p, v in self._control_points]

    def evaluate(self, progress: float) -> float:
        return float(self.evaluate_array(np.array([progress], dtype=float))[0])

    def evaluate_array(self, progress_array) -> np.ndarray:
        if self._xp is None or self._yp is None:
            return np.full_like(progress_array, self._default_value, dtype=float)
        progress = np.asarray(progress_array, dtype=float)
        if progress.size == 0:
            return np.array([], dtype=float)
        progress_wrapped = np.mod(progress, 1.0)
        # handle exact 1.0 by bringing back to 0 so np.interp wraps correctly
        progress_wrapped = np.where(np.isclose(progress_wrapped, 1.0), 0.0, progress_wrapped)
        values = np.interp(progress_wrapped, self._xp, self._yp)
        return values

    def _rebuild(self) -> None:
        ctrl = self._control_points
        if not ctrl:
            self._xp = np.array([0.0, 1.0])
            self._yp = np.array([self._default_value, self._default_value])
        else:
            xp, yp = zip(*ctrl)
            self._xp = np.array(xp, dtype=float)
            self._yp = np.array(yp, dtype=float)
