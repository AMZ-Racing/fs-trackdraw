from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PointList = List[Tuple[float, float]]


@dataclass
class BackgroundSpec:
    kind: str = "location"  # "location" or "grid"
    location_name: str = ""
    px_per_m: float = 10.0
    grid_width_px: int = 1600
    grid_height_px: int = 1200
    grid_spacing_m: float = 1.0


@dataclass
class LocationConfig:
    name: str
    px_per_m: float
    sat_img_path: str
    config_path: str


@dataclass
class GeneratorSettings:
    seed: int = 7
    complexity: str = "balanced"
    attempts: int = 80
    min_clearance_m: float = 1.0
    timeout_s: float = 2.5


@dataclass
class RuleSettings:
    enabled: bool = True
    threshold: Optional[float] = None
    severity: str = "error"


@dataclass
class ValidationIssue:
    rule_id: str
    severity: str
    summary: str
    detail: str


@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0


@dataclass
class TrackGeometry:
    centerline: Optional[np.ndarray] = None
    left_boundary: Optional[np.ndarray] = None
    right_boundary: Optional[np.ndarray] = None
    generated_left_cones: Optional[np.ndarray] = None
    generated_right_cones: Optional[np.ndarray] = None
    left_cones: Optional[np.ndarray] = None
    right_cones: Optional[np.ndarray] = None
    curvature_progress: Optional[np.ndarray] = None
    curvature_values: Optional[np.ndarray] = None
    track_length_m: float = 0.0
    min_radius_m: float = float("inf")

    @property
    def has_track(self) -> bool:
        return (
            self.centerline is not None
            and self.left_boundary is not None
            and self.right_boundary is not None
            and self.left_cones is not None
            and self.right_cones is not None
        )


@dataclass
class TrackOverlay:
    source_path: str = ""
    left_cones_local_m: PointList = field(default_factory=list)
    right_cones_local_m: PointList = field(default_factory=list)
    control_points_local_m: PointList = field(default_factory=list)
    center_x_px: float = 0.0
    center_y_px: float = 0.0
    rotation_deg: float = 0.0
    locked: bool = False
    preserve_cones: bool = False

    @property
    def has_data(self) -> bool:
        return bool(self.left_cones_local_m or self.right_cones_local_m or self.control_points_local_m)


@dataclass
class ProjectState:
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    allowed_area_points: PointList = field(default_factory=list)
    control_points: PointList = field(default_factory=list)
    track_mode: str = "manual"
    generator_settings: GeneratorSettings = field(default_factory=GeneratorSettings)
    rules: Dict[str, RuleSettings] = field(default_factory=dict)
    width_control_points: PointList = field(default_factory=list)
    cone_spacing_control_points: PointList = field(default_factory=list)
    left_cone_overrides: PointList = field(default_factory=list)
    right_cone_overrides: PointList = field(default_factory=list)
    imported_track: TrackOverlay = field(default_factory=TrackOverlay)
    project_path: str = ""
