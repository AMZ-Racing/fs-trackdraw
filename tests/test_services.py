import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from parameter_function import ParameterFunction
from trackdraw.configuration import discover_location_configs, load_rule_settings, load_track_defaults
from trackdraw.exporter import transform_points_to_export_frame
from trackdraw.generator import generate_control_points
from trackdraw.geometry import build_track_geometry
from trackdraw.importer import load_track_csv
from trackdraw.models import BackgroundSpec, GeneratorSettings, ProjectState, TrackOverlay
from trackdraw.project_io import load_project, save_project
from trackdraw.validation import build_default_rules, validate_track


class ConfigurationTests(unittest.TestCase):
    def test_load_defaults_and_locations(self):
        defaults = load_track_defaults(os.path.join(ROOT, "config", "track_config.yaml"))
        self.assertIn("track_width", defaults)
        self.assertGreater(defaults["track_width"], 0.0)

        locations = discover_location_configs(os.path.join(ROOT, "location_images"))
        self.assertGreaterEqual(len(locations), 1)
        self.assertTrue(any(location.name == defaults["standard_location"] for location in locations))

    def test_load_rule_settings(self):
        rules = load_rule_settings(os.path.join(ROOT, "config", "validation_rules.yaml"))
        self.assertIn("min_track_width", rules)
        self.assertTrue(rules["min_track_width"].enabled)


class ParameterFunctionTests(unittest.TestCase):
    def test_exact_endpoint_is_preserved(self):
        fn = ParameterFunction(3.0, name="Track Width")
        fn.set_control_points([(0.0, 3.0), (0.3, 4.2), (1.0, 3.0)])

        control_points = fn.get_control_points()
        self.assertEqual(control_points[0][0], 0.0)
        self.assertEqual(control_points[-1][0], 1.0)
        self.assertAlmostEqual(fn.evaluate(1.0), fn.evaluate(0.0))


class ExportTests(unittest.TestCase):
    def test_export_frame_preserves_first_point_as_origin(self):
        centerline = [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)]
        points = [(10.0, 10.0), (20.0, 20.0)]
        transformed = transform_points_to_export_frame(centerline, points, px_per_m=10.0)
        self.assertAlmostEqual(transformed[0][0], 0.0)
        self.assertAlmostEqual(transformed[0][1], 0.0)
        self.assertAlmostEqual(transformed[1][0], 1.0)
        self.assertAlmostEqual(transformed[1][1], 1.0)


class ImportTests(unittest.TestCase):
    def test_load_track_csv_centers_imported_points_and_creates_control_points(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "track.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("tag,x,y\n")
                handle.write("blue,1.0,1.0\n")
                handle.write("yellow,5.0,1.0\n")
                handle.write("blue,1.0,3.0\n")
                handle.write("yellow,5.0,3.0\n")
                handle.write("blue,2.0,4.0\n")
                handle.write("yellow,6.0,4.0\n")
                handle.write("blue,0.0,2.0\n")
                handle.write("yellow,4.0,2.0\n")

            imported = load_track_csv(csv_path)

        all_points = np.asarray(imported.left_cones_local_m + imported.right_cones_local_m, dtype=float)
        self.assertEqual(len(imported.left_cones_local_m), 4)
        self.assertEqual(len(imported.right_cones_local_m), 4)
        self.assertGreaterEqual(len(imported.control_points_local_m), 8)
        self.assertGreater(imported.track_width_m, 0.0)
        self.assertGreater(imported.cone_spacing_m, 0.0)
        np.testing.assert_allclose(np.mean(all_points, axis=0), np.zeros(2), atol=1e-9)


class ProjectRoundTripTests(unittest.TestCase):
    def test_project_round_trip(self):
        state = ProjectState(
            background=BackgroundSpec(kind="grid", px_per_m=8.0, grid_spacing_m=0.5),
            allowed_area_points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)],
            control_points=[(10.0, 10.0), (20.0, 20.0), (30.0, 20.0), (40.0, 10.0)],
            track_mode="manual",
            rules=build_default_rules(),
            imported_track=TrackOverlay(
                source_path="trackfiles_csv/test.csv",
                left_cones_local_m=[(-1.0, 0.0)],
                right_cones_local_m=[(1.0, 0.0)],
                control_points_local_m=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)],
                center_x_px=250.0,
                center_y_px=320.0,
                rotation_deg=15.0,
                locked=True,
                preserve_cones=True,
            ),
        )
        state.generator_settings = GeneratorSettings(seed=11, complexity="balanced", attempts=20, min_clearance_m=0.4, timeout_s=1.2)
        state.width_control_points = [(0.0, 3.0), (1.0, 3.0)]
        state.cone_spacing_control_points = [(0.0, 3.5), (1.0, 3.5)]
        state.left_cone_overrides = [(1.0, 2.0)]
        state.right_cone_overrides = [(3.0, 4.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.trackdraw.yaml")
            save_project(path, state)
            loaded = load_project(path)

        self.assertEqual(loaded.background.kind, "grid")
        self.assertEqual(loaded.background.grid_spacing_m, 0.5)
        self.assertEqual(loaded.allowed_area_points[0], (0.0, 0.0))
        self.assertEqual(loaded.left_cone_overrides[0], (1.0, 2.0))
        self.assertAlmostEqual(loaded.generator_settings.timeout_s, 1.2)
        self.assertEqual(loaded.imported_track.source_path, "trackfiles_csv/test.csv")
        self.assertAlmostEqual(loaded.imported_track.rotation_deg, 15.0)
        self.assertTrue(loaded.imported_track.locked)
        self.assertTrue(loaded.imported_track.preserve_cones)
        self.assertEqual(len(loaded.imported_track.control_points_local_m), 4)
        self.assertIn("min_track_width", loaded.rules)


class ValidationTests(unittest.TestCase):
    def test_valid_track_geometry_passes_relaxed_rules(self):
        width = ParameterFunction(3.0, name="Track Width")
        spacing = ParameterFunction(3.5, name="Cone Spacing")
        control_points = [(140.0, 120.0), (360.0, 120.0), (360.0, 300.0), (140.0, 300.0)]
        geometry = build_track_geometry(control_points, px_per_m=10.0, n_points_midline=250, width_function=width, cone_spacing_function=spacing)
        rules = build_default_rules()
        rules["min_track_length"].threshold = 0.0
        rules["min_clearance"].threshold = 0.1
        allowed_area = [(60.0, 60.0), (440.0, 60.0), (440.0, 360.0), (60.0, 360.0)]
        result = validate_track(allowed_area, geometry, 10.0, rules)
        self.assertTrue(geometry.has_track)
        self.assertGreater(geometry.min_radius_m, 5.0)
        self.assertEqual(result.error_count, 0)


class GeneratorTests(unittest.TestCase):
    def test_generator_is_deterministic_for_same_seed(self):
        width = ParameterFunction(3.0, name="Track Width")
        spacing = ParameterFunction(3.5, name="Cone Spacing")
        rules = build_default_rules()
        rules["min_track_length"].threshold = 0.0
        rules["min_clearance"].threshold = 0.1
        area = [(40.0, 40.0), (760.0, 40.0), (760.0, 560.0), (40.0, 560.0)]
        settings = GeneratorSettings(seed=17, complexity="balanced", attempts=60, min_clearance_m=0.4, timeout_s=2.0)

        first = generate_control_points(area, 10.0, width, spacing, 250, rules, settings)
        second = generate_control_points(area, 10.0, width, spacing, 250, rules, settings)

        self.assertTrue(first.control_points)
        self.assertEqual(first.control_points, second.control_points)


if __name__ == "__main__":
    unittest.main()
