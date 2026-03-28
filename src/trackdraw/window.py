from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from function_editor_qt import FunctionEditorDialog
from parameter_function import ParameterFunction

from .canvas import TrackCanvas
from .configuration import DEFAULT_RULES_CONFIG, default_background, discover_location_configs, load_rule_settings, load_track_defaults
from .dialogs import RulesEditorDialog
from .exporter import export_track_csv
from .generator import generate_control_points
from .geometry import array_to_points, build_track_geometry, points_to_array, set_cone_overrides, transform_points
from .importer import load_track_csv
from .models import BackgroundSpec, ProjectState, TrackGeometry, TrackOverlay
from .project_io import load_project, save_project
from .validation import RULE_DEFINITIONS, ValidationResult, build_default_rules, validate_track


APP_LOGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "TrackDraw_Logo.png"))


class TrackDrawWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrackDraw")
        icon = QIcon(APP_LOGO_PATH)
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.resize(1440, 920)
        self.setMinimumSize(1100, 760)

        self.defaults = load_track_defaults("config/track_config.yaml")
        self.location_configs = discover_location_configs("location_images")
        self.locations_by_name = {item.name: item for item in self.location_configs}
        self.default_rule_settings = build_default_rules(load_rule_settings(DEFAULT_RULES_CONFIG))

        self.track_width_function = ParameterFunction(self.defaults["track_width"], name="Track Width")
        self.cone_spacing_function = ParameterFunction(self.defaults["cone_distance"], name="Cone Spacing")

        self.project_state = ProjectState(
            background=default_background(self.defaults["standard_location"], self.location_configs),
            rules=copy.deepcopy(self.default_rule_settings),
        )
        self.project_state.width_control_points = self.track_width_function.get_control_points()
        self.project_state.cone_spacing_control_points = self.cone_spacing_function.get_control_points()

        self.track_geometry = TrackGeometry()
        self.validation_result = ValidationResult()

        self.edit_target = "area"
        self.edit_mode = "add"
        self.dragging_kind: Optional[str] = None
        self.dragging_index: Optional[int] = None
        self.dragging_cone_side: Optional[str] = None
        self._transform_drag_start_point: Optional[np.ndarray] = None
        self._transform_drag_start_center: Optional[np.ndarray] = None
        self._transform_drag_start_rotation: Optional[float] = None
        self._transform_drag_start_angle: Optional[float] = None
        self.suppress_ui = False
        self._drag_rebuild_timer = QTimer(self)
        self._drag_rebuild_timer.setSingleShot(True)
        self._drag_rebuild_timer.timeout.connect(self._rebuild_track)

        self._build_ui()
        self._apply_background_to_canvas(reset_area_if_empty=True)
        self._set_full_allowed_area_if_empty()
        self._rebuild_track()
        self.statusBar().showMessage("Ready")

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        self.canvas = TrackCanvas(self)
        self.canvas.setMinimumSize(700, 600)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.canvas)

        self.sidebar_scroll = QScrollArea(self)
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(420)
        splitter.addWidget(self.sidebar_scroll)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1080, 440])

        sidebar = QWidget()
        self.sidebar_scroll.setWidget(sidebar)
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_layout.setSpacing(10)

        self._build_project_section()
        self._build_background_section()
        self._build_editor_section()
        self._build_track_section()
        self._build_validation_section()
        self._build_actions_section()
        self._build_stats_section()
        self._build_logo_section()
        self.sidebar_layout.addStretch(1)

    def _build_project_section(self):
        box = QGroupBox("Project")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self.new_project_button = QPushButton("New")
        self.new_project_button.clicked.connect(self._new_project)
        self.open_project_button = QPushButton("Open")
        self.open_project_button.clicked.connect(self._open_project)
        self.save_project_button = QPushButton("Save")
        self.save_project_button.clicked.connect(self._save_project)
        self.save_as_project_button = QPushButton("Save As")
        self.save_as_project_button.clicked.connect(self._save_project_as)
        row.addWidget(self.new_project_button)
        row.addWidget(self.open_project_button)
        row.addWidget(self.save_project_button)
        row.addWidget(self.save_as_project_button)

        self.help_button = QPushButton("Help")
        self.help_button.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        self.help_button.clicked.connect(self._show_help)
        row.addWidget(self.help_button)
        layout.addLayout(row)

        self.project_path_label = QLabel("Project: unsaved")
        self.project_path_label.setWordWrap(True)
        layout.addWidget(self.project_path_label)
        self.sidebar_layout.addWidget(box)

    def _build_background_section(self):
        box = QGroupBox("Background")
        layout = QFormLayout(box)

        self.background_kind_combo = QComboBox()
        self.background_kind_combo.addItem("Location Image", "location")
        self.background_kind_combo.addItem("Metric Grid", "grid")
        self.background_kind_combo.currentIndexChanged.connect(self._on_background_kind_changed)
        layout.addRow("Type", self.background_kind_combo)

        self.location_combo = QComboBox()
        for location in self.location_configs:
            self.location_combo.addItem(location.name, location.name)
        self.location_combo.currentIndexChanged.connect(self._on_location_changed)
        layout.addRow("Location", self.location_combo)

        self.grid_px_per_m_entry = QLineEdit(f"{self.project_state.background.px_per_m:.3f}")
        self.grid_px_per_m_entry.editingFinished.connect(self._on_grid_scale_changed)
        layout.addRow("Grid px/m", self.grid_px_per_m_entry)

        self.grid_spacing_entry = QLineEdit(f"{self.project_state.background.grid_spacing_m:.3f}")
        self.grid_spacing_entry.editingFinished.connect(self._on_grid_spacing_changed)
        layout.addRow("Grid spacing (m)", self.grid_spacing_entry)

        self.fill_area_button = QPushButton("Fill Allowed Area")
        self.fill_area_button.clicked.connect(self._fill_allowed_area)
        layout.addRow(self.fill_area_button)

        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.canvas.reset_view)
        layout.addRow(self.reset_view_button)

        self.sidebar_layout.addWidget(box)

    def _build_editor_section(self):
        box = QGroupBox("Editor")
        layout = QVBoxLayout(box)

        target_row = QHBoxLayout()
        self.target_group = QButtonGroup(self)
        self.area_button = QRadioButton("Allowed Area")
        self.track_button = QRadioButton("Track")
        self.cone_button = QRadioButton("Cones")
        self.area_button.setChecked(True)
        for button, target in ((self.area_button, "area"), (self.track_button, "track"), (self.cone_button, "cones")):
            self.target_group.addButton(button)
            button.clicked.connect(lambda checked, value=target: self._set_edit_target(value))
            target_row.addWidget(button)
        layout.addLayout(target_row)

        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.add_mode_button = QRadioButton("Add")
        self.move_mode_button = QRadioButton("Move")
        self.remove_mode_button = QRadioButton("Remove")
        self.add_mode_button.setChecked(True)
        for button, mode in ((self.add_mode_button, "add"), (self.move_mode_button, "move"), (self.remove_mode_button, "remove")):
            self.mode_group.addButton(button)
            button.clicked.connect(lambda checked, value=mode: self._set_edit_mode(value))
            mode_row.addWidget(button)
        layout.addLayout(mode_row)

        self.mode_hint_label = QLabel()
        self.mode_hint_label.setWordWrap(True)
        layout.addWidget(self.mode_hint_label)
        self.sidebar_layout.addWidget(box)
        self._update_mode_hint()

    def _build_track_section(self):
        box = QGroupBox("Track")
        layout = QFormLayout(box)

        mode_row = QHBoxLayout()
        self.manual_mode_button = QRadioButton("Manual")
        self.auto_mode_button = QRadioButton("Auto")
        self.manual_mode_button.setChecked(True)
        self.manual_mode_button.toggled.connect(self._on_track_mode_changed)
        self.auto_mode_button.toggled.connect(self._on_track_mode_changed)
        mode_row.addWidget(self.manual_mode_button)
        mode_row.addWidget(self.auto_mode_button)
        layout.addRow("Creation", mode_row)

        self.track_width_entry = QLineEdit(f"{self.track_width_function.evaluate(0.0):.3f}")
        self.track_width_entry.editingFinished.connect(self._update_track_width_constant)
        layout.addRow("Track width (m)", self.track_width_entry)

        self.track_width_advanced_button = QPushButton("Edit Width Profile")
        self.track_width_advanced_button.clicked.connect(self._open_track_width_editor)
        layout.addRow(self.track_width_advanced_button)

        self.cone_spacing_entry = QLineEdit(f"{self.cone_spacing_function.evaluate(0.0):.3f}")
        self.cone_spacing_entry.editingFinished.connect(self._update_cone_spacing_constant)
        layout.addRow("Cone spacing (m)", self.cone_spacing_entry)

        self.cone_spacing_advanced_button = QPushButton("Edit Spacing Profile")
        self.cone_spacing_advanced_button.clicked.connect(self._open_cone_spacing_editor)
        layout.addRow(self.cone_spacing_advanced_button)

        self.generator_seed_entry = QLineEdit(str(self.project_state.generator_settings.seed))
        layout.addRow("Generator seed", self.generator_seed_entry)

        self.generator_complexity_combo = QComboBox()
        self.generator_complexity_combo.addItem("Simple", "simple")
        self.generator_complexity_combo.addItem("Balanced", "balanced")
        self.generator_complexity_combo.addItem("Complex", "complex")
        layout.addRow("Complexity", self.generator_complexity_combo)

        self.generator_clearance_entry = QLineEdit(f"{self.project_state.generator_settings.min_clearance_m:.3f}")
        layout.addRow("Min clearance (m)", self.generator_clearance_entry)

        self.generator_timeout_entry = QLineEdit(f"{self.project_state.generator_settings.timeout_s:.3f}")
        layout.addRow("Timeout (s)", self.generator_timeout_entry)

        self.generate_button = QPushButton("Generate Track")
        self.generate_button.clicked.connect(self._generate_track)
        layout.addRow(self.generate_button)

        self.sidebar_layout.addWidget(box)
        self._on_track_mode_changed()

    def _build_validation_section(self):
        box = QGroupBox("Validation")
        layout = QVBoxLayout(box)

        self.edit_rules_button = QPushButton("Edit Rules")
        self.edit_rules_button.clicked.connect(self._edit_rules)
        layout.addWidget(self.edit_rules_button)

        self.validation_summary_label = QLabel("Validation: not run")
        layout.addWidget(self.validation_summary_label)

        self.validation_list = QListWidget()
        self.validation_list.setMinimumHeight(180)
        self.validation_list.setWordWrap(True)
        self.validation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.validation_list)

        self.sidebar_layout.addWidget(box)

    def _build_actions_section(self):
        box = QGroupBox("Actions")
        layout = QVBoxLayout(box)

        action_row = QHBoxLayout()
        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self._export_csv)
        action_row.addWidget(self.export_button)

        self.load_csv_button = QPushButton("Load CSV")
        self.load_csv_button.clicked.connect(self._load_track_csv)
        action_row.addWidget(self.load_csv_button)

        self.clear_csv_button = QPushButton("Clear CSV")
        self.clear_csv_button.clicked.connect(self._clear_imported_track)
        action_row.addWidget(self.clear_csv_button)
        layout.addLayout(action_row)

        self.imported_track_label = QLabel("Loaded CSV: none")
        self.imported_track_label.setWordWrap(True)
        layout.addWidget(self.imported_track_label)

        self.imported_track_state_label = QLabel("Imported track: none")
        self.imported_track_state_label.setWordWrap(True)
        layout.addWidget(self.imported_track_state_label)

        self.toggle_shape_lock_button = QPushButton("Unlock Shape")
        self.toggle_shape_lock_button.clicked.connect(self._toggle_imported_track_lock)
        layout.addWidget(self.toggle_shape_lock_button)

        self.overlay_hint_label = QLabel(
            "Loaded CSV tracks create a centerline immediately. While shape is locked, use the on-canvas handles to translate or rotate the whole track."
        )
        self.overlay_hint_label.setWordWrap(True)
        layout.addWidget(self.overlay_hint_label)

        self.sidebar_layout.addWidget(box)

    def _build_stats_section(self):
        box = QGroupBox("Stats")
        layout = QVBoxLayout(box)

        self.track_length_label = QLabel("Track length: --")
        self.min_radius_label = QLabel("Min radius: --")
        self.cone_count_label = QLabel("Cones: --")
        self.background_info_label = QLabel("Scale: --")

        layout.addWidget(self.track_length_label)
        layout.addWidget(self.min_radius_label)
        layout.addWidget(self.cone_count_label)
        layout.addWidget(self.background_info_label)
        self.sidebar_layout.addWidget(box)

    def _build_logo_section(self):
        self.logo_label = QLabel()
        self.logo_pixmap = QPixmap(APP_LOGO_PATH)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(self.logo_label)
        self._update_logo_size()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_logo_size()

    def _update_logo_size(self):
        if self.logo_pixmap.isNull():
            return
        max_width = 220
        scaled = self.logo_pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
        self.logo_label.setPixmap(scaled)
        self.logo_label.setFixedHeight(scaled.height() + 6)

    def _set_edit_target(self, target: str):
        self.edit_target = target
        self._update_mode_hint()
        self._update_canvas()

    def _set_edit_mode(self, mode: str):
        self.edit_mode = mode
        self._update_mode_hint()

    def _update_mode_hint(self):
        imported = self.project_state.imported_track
        if self.edit_target == "track" and imported.has_data and imported.locked:
            self.mode_hint_label.setText(
                "Track shape is locked. Drag the square to move freely, the red/blue arrows to constrain translation, or the purple handle to rotate the whole track."
            )
            return
        if self.edit_target == "cones" and imported.preserve_cones:
            self.mode_hint_label.setText(
                "Imported cone positions are being preserved. Unlock and reshape the centerline to regenerate editable cones."
            )
            return

        descriptions = {
            ("area", "add"): "Click to add allowed-area vertices. Use Fill Allowed Area for a quick starting polygon.",
            ("area", "move"): "Click near an allowed-area vertex, then drag to reposition it.",
            ("area", "remove"): "Click an allowed-area vertex to remove it.",
            ("track", "add"): "Click to add manual centerline control points.",
            ("track", "move"): "Click near a control point, then drag to reshape the track.",
            ("track", "remove"): "Click a control point to remove it.",
            ("cones", "add"): "Click a cone to insert a new cone halfway to the next cone on that side.",
            ("cones", "move"): "Click near a cone, then drag to override its position.",
            ("cones", "remove"): "Click a cone to remove it from the exported layout.",
        }
        self.mode_hint_label.setText(descriptions.get((self.edit_target, self.edit_mode), ""))

    def _update_editor_controls(self):
        imported = self.project_state.imported_track
        cones_editable = not imported.preserve_cones
        self.cone_button.setEnabled(cones_editable)
        if not cones_editable and self.edit_target == "cones":
            self.edit_target = "track"
            self.track_button.setChecked(True)
        self._update_mode_hint()

    def _copy_default_rules(self) -> Dict[str, object]:
        return {rule_id: copy.deepcopy(rule) for rule_id, rule in self.default_rule_settings.items()}

    def _new_project(self):
        if self.project_state.control_points or self.project_state.allowed_area_points:
            result = QMessageBox.question(
                self,
                "New Project",
                "Start a new project and discard the current editable state?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if result != QMessageBox.Yes:
                return

        self.track_width_function.set_constant(self.defaults["track_width"])
        self.cone_spacing_function.set_constant(self.defaults["cone_distance"])
        self.project_state = ProjectState(
            background=default_background(self.defaults["standard_location"], self.location_configs),
            rules=self._copy_default_rules(),
        )
        self.project_state.width_control_points = self.track_width_function.get_control_points()
        self.project_state.cone_spacing_control_points = self.cone_spacing_function.get_control_points()
        self.project_state.project_path = ""
        self.manual_mode_button.setChecked(True)
        self._apply_project_state_to_ui(reset_area_if_empty=True)
        self.statusBar().showMessage("Started a new project")

    def _open_project(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open TrackDraw Project", "", "TrackDraw YAML (*.trackdraw.yaml *.yaml)")
        if not filename:
            return
        state = load_project(filename)
        self.project_state = state
        self.track_width_function.set_control_points(state.width_control_points or [(0.0, self.defaults["track_width"]), (1.0, self.defaults["track_width"])])
        self.cone_spacing_function.set_control_points(state.cone_spacing_control_points or [(0.0, self.defaults["cone_distance"]), (1.0, self.defaults["cone_distance"])])
        if not state.rules:
            self.project_state.rules = self._copy_default_rules()
        self._apply_project_state_to_ui(reset_area_if_empty=False)
        self.statusBar().showMessage(f"Opened {filename}")

    def _save_project(self):
        if not self.project_state.project_path:
            self._save_project_as()
            return
        self._sync_project_state()
        save_project(self.project_state.project_path, self.project_state)
        self.statusBar().showMessage(f"Saved {self.project_state.project_path}")

    def _save_project_as(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save TrackDraw Project", self.project_state.project_path or "track.trackdraw.yaml", "TrackDraw YAML (*.trackdraw.yaml *.yaml)")
        if not filename:
            return
        if not filename.endswith(".yaml"):
            filename += ".trackdraw.yaml"
        self.project_state.project_path = filename
        self._sync_project_state()
        save_project(filename, self.project_state)
        self._update_project_path_label()
        self.statusBar().showMessage(f"Saved {filename}")

    def _sync_project_state(self):
        self.project_state.width_control_points = self.track_width_function.get_control_points()
        self.project_state.cone_spacing_control_points = self.cone_spacing_function.get_control_points()
        self.project_state.track_mode = "auto" if self.auto_mode_button.isChecked() else "manual"
        self.project_state.generator_settings.seed = self._read_int_entry(self.generator_seed_entry, self.project_state.generator_settings.seed)
        self.project_state.generator_settings.complexity = self.generator_complexity_combo.currentData()
        self.project_state.generator_settings.min_clearance_m = self._read_float_entry(self.generator_clearance_entry, self.project_state.generator_settings.min_clearance_m, minimum=0.0)
        self.project_state.generator_settings.timeout_s = self._read_float_entry(self.generator_timeout_entry, self.project_state.generator_settings.timeout_s, minimum=0.1)
        self.project_state.background.grid_spacing_m = self._read_float_entry(self.grid_spacing_entry, self.project_state.background.grid_spacing_m, minimum=0.1)
        if self.project_state.background.kind == "grid":
            self.project_state.background.px_per_m = self._read_float_entry(self.grid_px_per_m_entry, self.project_state.background.px_per_m, minimum=0.1)
        self._update_project_path_label()

    def _update_project_path_label(self):
        self.project_path_label.setText(f"Project: {self.project_state.project_path or 'unsaved'}")

    def _apply_project_state_to_ui(self, reset_area_if_empty: bool):
        self.suppress_ui = True
        background_kind = self.project_state.background.kind
        index = self.background_kind_combo.findData(background_kind)
        self.background_kind_combo.setCurrentIndex(max(index, 0))
        location_index = self.location_combo.findData(self.project_state.background.location_name)
        if location_index >= 0:
            self.location_combo.setCurrentIndex(location_index)
        self.grid_px_per_m_entry.setText(f"{self.project_state.background.px_per_m:.3f}")
        self.grid_spacing_entry.setText(f"{self.project_state.background.grid_spacing_m:.3f}")
        self.track_width_entry.setText(f"{self.track_width_function.evaluate(0.0):.3f}")
        self.cone_spacing_entry.setText(f"{self.cone_spacing_function.evaluate(0.0):.3f}")
        self.generator_seed_entry.setText(str(self.project_state.generator_settings.seed))
        self.generator_clearance_entry.setText(f"{self.project_state.generator_settings.min_clearance_m:.3f}")
        self.generator_timeout_entry.setText(f"{self.project_state.generator_settings.timeout_s:.3f}")
        complexity_index = self.generator_complexity_combo.findData(self.project_state.generator_settings.complexity)
        self.generator_complexity_combo.setCurrentIndex(max(complexity_index, 1))
        self.manual_mode_button.setChecked(self.project_state.track_mode != "auto")
        self.auto_mode_button.setChecked(self.project_state.track_mode == "auto")
        self.suppress_ui = False

        self._apply_background_to_canvas(reset_area_if_empty=reset_area_if_empty)
        self._set_full_allowed_area_if_empty()
        self._update_imported_track_controls()
        self._update_editor_controls()
        self._rebuild_track()
        self._update_project_path_label()

    def _apply_background_to_canvas(self, reset_area_if_empty: bool):
        background = self.project_state.background
        if background.kind == "location":
            location = self.locations_by_name.get(background.location_name)
            if location is None:
                self.project_state.background = BackgroundSpec(kind="grid", px_per_m=background.px_per_m, grid_spacing_m=background.grid_spacing_m)
                background = self.project_state.background
            else:
                self.project_state.background.px_per_m = location.px_per_m
                self.canvas.set_background(
                    "location",
                    location.px_per_m,
                    image_path=location.sat_img_path,
                    grid_spacing_m=background.grid_spacing_m,
                )
                self.background_info_label.setText(self._format_background_info())
                self._update_background_controls()
                self._update_imported_track_controls()
                if reset_area_if_empty:
                    self._set_full_allowed_area_if_empty()
                return

        self.canvas.set_background(
            "grid",
            self.project_state.background.px_per_m,
            width_px=self.project_state.background.grid_width_px,
            height_px=self.project_state.background.grid_height_px,
            grid_spacing_m=self.project_state.background.grid_spacing_m,
        )
        self.background_info_label.setText(self._format_background_info())
        self._update_background_controls()
        self._update_imported_track_controls()
        if reset_area_if_empty:
            self._set_full_allowed_area_if_empty()

    def _update_background_controls(self):
        is_location = self.project_state.background.kind == "location"
        self.location_combo.setEnabled(is_location)
        self.grid_px_per_m_entry.setEnabled(not is_location)
        self.grid_spacing_entry.setEnabled(not is_location)

    def _set_full_allowed_area_if_empty(self):
        if self.project_state.allowed_area_points:
            return
        self._fill_allowed_area()

    def _fill_allowed_area(self):
        bg_w, bg_h = self.canvas.background_dimensions()
        margin = 0.08 * min(bg_w, bg_h)
        self.project_state.allowed_area_points = [
            (margin, margin),
            (bg_w - margin, margin),
            (bg_w - margin, bg_h - margin),
            (margin, bg_h - margin),
        ]
        self._revalidate()
        self._update_canvas()
        self.statusBar().showMessage("Filled allowed area to background bounds")

    def _on_background_kind_changed(self):
        if self.suppress_ui:
            return
        new_kind = self.background_kind_combo.currentData()
        if new_kind == self.project_state.background.kind:
            return
        if not self._confirm_background_change():
            self.suppress_ui = True
            index = self.background_kind_combo.findData(self.project_state.background.kind)
            self.background_kind_combo.setCurrentIndex(max(index, 0))
            self.suppress_ui = False
            return

        if new_kind == "location":
            location_name = self.location_combo.currentData() or (self.location_configs[0].name if self.location_configs else "")
            px_per_m = self.locations_by_name.get(location_name).px_per_m if location_name in self.locations_by_name else self.project_state.background.px_per_m
            self.project_state.background = BackgroundSpec(
                kind="location",
                location_name=location_name,
                px_per_m=px_per_m,
                grid_spacing_m=self.project_state.background.grid_spacing_m,
            )
        else:
            px_per_m = self._read_float_entry(self.grid_px_per_m_entry, self.project_state.background.px_per_m, minimum=0.1)
            grid_spacing_m = self._read_float_entry(self.grid_spacing_entry, self.project_state.background.grid_spacing_m, minimum=0.1)
            self.project_state.background = BackgroundSpec(kind="grid", px_per_m=px_per_m, grid_spacing_m=grid_spacing_m)
        self.project_state.allowed_area_points = []
        self.project_state.control_points = []
        self.project_state.imported_track = TrackOverlay()
        self._clear_cone_overrides()
        self._apply_background_to_canvas(reset_area_if_empty=True)
        self._rebuild_track()

    def _on_location_changed(self):
        if self.suppress_ui or self.project_state.background.kind != "location":
            return
        location_name = self.location_combo.currentData()
        if not location_name or location_name == self.project_state.background.location_name:
            return
        if not self._confirm_background_change():
            self.suppress_ui = True
            index = self.location_combo.findData(self.project_state.background.location_name)
            self.location_combo.setCurrentIndex(max(index, 0))
            self.suppress_ui = False
            return
        location = self.locations_by_name[location_name]
        self.project_state.background = BackgroundSpec(
            kind="location",
            location_name=location.name,
            px_per_m=location.px_per_m,
            grid_spacing_m=self.project_state.background.grid_spacing_m,
        )
        self.project_state.allowed_area_points = []
        self.project_state.control_points = []
        self.project_state.imported_track = TrackOverlay()
        self._clear_cone_overrides()
        self._apply_background_to_canvas(reset_area_if_empty=True)
        self._rebuild_track()

    def _on_grid_scale_changed(self):
        if self.suppress_ui or self.project_state.background.kind != "grid":
            return
        new_scale = self._read_float_entry(self.grid_px_per_m_entry, self.project_state.background.px_per_m, minimum=0.1)
        self.project_state.background.px_per_m = new_scale
        self._apply_background_to_canvas(reset_area_if_empty=False)
        self._rebuild_track()

    def _on_grid_spacing_changed(self):
        if self.suppress_ui or self.project_state.background.kind != "grid":
            return
        new_spacing = self._read_float_entry(self.grid_spacing_entry, self.project_state.background.grid_spacing_m, minimum=0.1)
        self.project_state.background.grid_spacing_m = new_spacing
        self._apply_background_to_canvas(reset_area_if_empty=False)
        self._update_canvas()

    def _confirm_background_change(self) -> bool:
        if not (self.project_state.allowed_area_points or self.project_state.control_points or self.track_geometry.has_track or self.project_state.imported_track.has_data):
            return True
        result = QMessageBox.question(
            self,
            "Change Background",
            "Changing background resets the allowed area, current track geometry, and any imported CSV track. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return result == QMessageBox.Yes

    def _on_track_mode_changed(self):
        is_auto = self.auto_mode_button.isChecked()
        self.generate_button.setEnabled(is_auto)
        self.generator_seed_entry.setEnabled(is_auto)
        self.generator_complexity_combo.setEnabled(is_auto)
        self.generator_clearance_entry.setEnabled(is_auto)
        self.generator_timeout_entry.setEnabled(is_auto)
        self.project_state.track_mode = "auto" if is_auto else "manual"

    def _edit_rules(self):
        dialog = RulesEditorDialog(self, self.project_state.rules)
        if dialog.exec_():
            self.project_state.rules = dialog.get_rules()
            self._revalidate()
            self.statusBar().showMessage("Updated validation rules")

    def _open_track_width_editor(self):
        old_points = self.track_width_function.get_control_points()
        dialog = FunctionEditorDialog(
            self,
            self.track_width_function,
            "Track Width Profile",
            units="m",
            reference=self._curvature_reference(),
        )
        if dialog.exec_():
            if old_points != self.track_width_function.get_control_points() and not self._confirm_discard_cone_overrides("Changing track width will regenerate boundaries and cones."):
                self.track_width_function.set_control_points(old_points)
                return
            self.track_width_entry.setText(f"{self.track_width_function.evaluate(0.0):.3f}")
            self._discard_preserved_imported_cones()
            self._rebuild_track()

    def _open_cone_spacing_editor(self):
        old_points = self.cone_spacing_function.get_control_points()
        dialog = FunctionEditorDialog(
            self,
            self.cone_spacing_function,
            "Cone Spacing Profile",
            units="m",
            reference=self._curvature_reference(),
        )
        if dialog.exec_():
            if old_points != self.cone_spacing_function.get_control_points() and not self._confirm_discard_cone_overrides("Changing cone spacing will regenerate cone positions."):
                self.cone_spacing_function.set_control_points(old_points)
                return
            self.cone_spacing_entry.setText(f"{self.cone_spacing_function.evaluate(0.0):.3f}")
            self._discard_preserved_imported_cones()
            self._rebuild_track()

    def _curvature_reference(self):
        if self.track_geometry.curvature_progress is None or self.track_geometry.curvature_values is None:
            return None
        return {
            "label": "Curvature (1/m)",
            "progress": self.track_geometry.curvature_progress,
            "values": self.track_geometry.curvature_values,
        }

    def _update_track_width_constant(self):
        old_points = self.track_width_function.get_control_points()
        new_value = self._read_float_entry(self.track_width_entry, self.track_width_function.evaluate(0.0), minimum=0.1)
        if np.isclose(new_value, self.track_width_function.evaluate(0.0)):
            self.track_width_entry.setText(f"{new_value:.3f}")
            return
        if not self._confirm_discard_cone_overrides("Changing track width will regenerate boundaries and cones."):
            self.track_width_function.set_control_points(old_points)
            self.track_width_entry.setText(f"{self.track_width_function.evaluate(0.0):.3f}")
            return
        self.track_width_function.set_constant(new_value)
        self.track_width_entry.setText(f"{new_value:.3f}")
        self._discard_preserved_imported_cones()
        self._rebuild_track()

    def _update_cone_spacing_constant(self):
        old_points = self.cone_spacing_function.get_control_points()
        new_value = self._read_float_entry(self.cone_spacing_entry, self.cone_spacing_function.evaluate(0.0), minimum=0.1)
        if np.isclose(new_value, self.cone_spacing_function.evaluate(0.0)):
            self.cone_spacing_entry.setText(f"{new_value:.3f}")
            return
        if not self._confirm_discard_cone_overrides("Changing cone spacing will regenerate cone positions."):
            self.cone_spacing_function.set_control_points(old_points)
            self.cone_spacing_entry.setText(f"{self.cone_spacing_function.evaluate(0.0):.3f}")
            return
        self.cone_spacing_function.set_constant(new_value)
        self.cone_spacing_entry.setText(f"{new_value:.3f}")
        self._discard_preserved_imported_cones()
        self._rebuild_track()

    def _generate_track(self):
        if len(self.project_state.allowed_area_points) < 3:
            QMessageBox.warning(self, "Generate Track", "Define the allowed area polygon first.")
            return
        if not self._confirm_discard_cone_overrides("Generating a new track replaces the current cone layout."):
            return
        self._sync_project_state()
        self.statusBar().showMessage("Generating track...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            result = generate_control_points(
                allowed_area_points=self.project_state.allowed_area_points,
                px_per_m=self.project_state.background.px_per_m,
                width_function=self.track_width_function,
                cone_spacing_function=self.cone_spacing_function,
                n_points_midline=self.defaults["n_points_midline"],
                rules=self.project_state.rules,
                settings=self.project_state.generator_settings,
            )
        finally:
            QApplication.restoreOverrideCursor()
        if not result.control_points:
            QMessageBox.warning(self, "Generate Track", result.message)
            self.statusBar().showMessage("Track generation did not produce a usable result")
            return
        self.project_state.imported_track = TrackOverlay()
        self.project_state.control_points = result.control_points
        self._update_imported_track_controls()
        self._update_editor_controls()
        self._rebuild_track()
        if result.succeeded:
            self.statusBar().showMessage("Generated a valid track")
        else:
            QMessageBox.warning(self, "Generate Track", f"Best candidate loaded with warnings\n{result.message}")

    def _export_csv(self):
        if not self.track_geometry.has_track or self.track_geometry.left_cones is None or self.track_geometry.right_cones is None:
            QMessageBox.warning(self, "Export CSV", "No track is available to export.")
            return
        if self.validation_result.issues:
            result = QMessageBox.question(
                self,
                "Export With Validation Findings",
                f"Export anyway with {self.validation_result.error_count} errors and {self.validation_result.warning_count} warnings?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if result != QMessageBox.Yes:
                return
        filename, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV files (*.csv)")
        if not filename:
            return
        export_track_csv(
            filename,
            self.track_geometry.centerline,
            self.track_geometry.left_cones,
            self.track_geometry.right_cones,
            self.project_state.background.px_per_m,
        )
        self.statusBar().showMessage(f"Exported {filename}")

    def has_active_drag(self) -> bool:
        return self.dragging_kind is not None

    def handle_canvas_click(self, pos: QPointF, handle_kind: Optional[str] = None):
        point = (float(pos.x()), float(pos.y()))
        if handle_kind and self.edit_target == "track" and self._has_locked_imported_track():
            self._start_track_transform_drag(handle_kind, point)
            return
        if self.edit_target == "area":
            self._handle_area_click(point)
        elif self.edit_target == "track":
            self._handle_track_click(point)
        else:
            self._handle_cone_click(point)

    def handle_canvas_drag(self, pos: QPointF):
        point = (float(pos.x()), float(pos.y()))
        if self.dragging_kind in {"track_translate_free", "track_translate_x", "track_translate_y", "track_rotate"}:
            self._apply_track_transform_drag(point)
        elif self.dragging_kind == "area" and self.dragging_index is not None:
            self.project_state.allowed_area_points[self.dragging_index] = point
            self._revalidate()
            self._update_canvas()
        elif self.dragging_kind == "track" and self.dragging_index is not None:
            self.project_state.control_points[self.dragging_index] = point
            if self.project_state.imported_track.has_data:
                self._capture_control_points_into_imported_track()
            self._update_canvas()
            self._schedule_track_rebuild()
        elif self.dragging_kind == "cone" and self.dragging_index is not None and self.dragging_cone_side is not None:
            cones = self._editable_cone_array(self.dragging_cone_side)
            if cones.shape[0] == 0 or self.dragging_index >= cones.shape[0]:
                return
            cones[self.dragging_index] = np.array(point, dtype=float)
            self._set_editable_cone_array(self.dragging_cone_side, cones)
            self._store_cone_overrides()
            self._revalidate()
            self._update_canvas()

    def handle_canvas_release(self, pos: QPointF):
        if self.dragging_kind == "track":
            self._flush_scheduled_track_rebuild()
        self.dragging_kind = None
        self.dragging_index = None
        self.dragging_cone_side = None
        self._transform_drag_start_point = None
        self._transform_drag_start_center = None
        self._transform_drag_start_rotation = None
        self._transform_drag_start_angle = None

    def _handle_area_click(self, point: Tuple[float, float]):
        if self.edit_mode == "add":
            self.project_state.allowed_area_points.append(point)
        elif self.edit_mode == "remove":
            index = self._find_near_point(self.project_state.allowed_area_points, point)
            if index is not None:
                del self.project_state.allowed_area_points[index]
        else:
            index = self._find_near_point(self.project_state.allowed_area_points, point, threshold=18.0)
            if index is not None:
                self.dragging_kind = "area"
                self.dragging_index = index
                self.project_state.allowed_area_points[index] = point
        self._revalidate()
        self._update_canvas()

    def _handle_track_click(self, point: Tuple[float, float]):
        if self.auto_mode_button.isChecked():
            self.manual_mode_button.setChecked(True)
        if self._has_locked_imported_track():
            self.statusBar().showMessage("Use the on-canvas handles to move or rotate the locked track.")
            return
        if not self._confirm_discard_cone_overrides("Editing the centerline will regenerate boundaries and cones."):
            return
        self._discard_preserved_imported_cones()
        if self.edit_mode == "add":
            self.project_state.control_points.append(point)
            if self.project_state.imported_track.has_data:
                self._capture_control_points_into_imported_track()
            self._rebuild_track()
        elif self.edit_mode == "remove":
            index = self._find_near_point(self.project_state.control_points, point)
            if index is not None:
                del self.project_state.control_points[index]
                if self.project_state.imported_track.has_data:
                    self._capture_control_points_into_imported_track()
                self._rebuild_track()
        else:
            index = self._find_near_point(self.project_state.control_points, point, threshold=18.0)
            if index is not None:
                self.dragging_kind = "track"
                self.dragging_index = index
                self.project_state.control_points[index] = point
                if self.project_state.imported_track.has_data:
                    self._capture_control_points_into_imported_track()
                self._update_canvas()
                self._schedule_track_rebuild()

    def _handle_cone_click(self, point: Tuple[float, float]):
        if self.project_state.imported_track.preserve_cones:
            self.statusBar().showMessage("Imported cone positions are being preserved. Unlock and reshape the centerline before editing cones.")
            return
        selection = self._find_near_cone(point)
        if selection is None:
            return
        side, index = selection
        cones = self._editable_cone_array(side)
        if cones.shape[0] == 0:
            return
        if self.edit_mode == "add":
            next_index = (index + 1) % cones.shape[0]
            midpoint = 0.5 * (cones[index] + cones[next_index])
            cones = np.insert(cones, next_index, midpoint, axis=0)
            self._set_editable_cone_array(side, cones)
            self._store_cone_overrides()
        elif self.edit_mode == "remove":
            cones = np.delete(cones, index, axis=0)
            self._set_editable_cone_array(side, cones)
            self._store_cone_overrides()
        else:
            self.dragging_kind = "cone"
            self.dragging_index = index
            self.dragging_cone_side = side
        self._revalidate()
        self._update_canvas()

    def _schedule_track_rebuild(self):
        if not self._drag_rebuild_timer.isActive():
            self._drag_rebuild_timer.start(20)

    def _flush_scheduled_track_rebuild(self):
        if self._drag_rebuild_timer.isActive():
            self._drag_rebuild_timer.stop()
            self._rebuild_track()

    def _overlay_center_default(self) -> Tuple[float, float]:
        area = points_to_array(self.project_state.allowed_area_points)
        if area.size:
            centroid = np.mean(area, axis=0)
            return float(centroid[0]), float(centroid[1])
        bg_w, bg_h = self.canvas.background_dimensions()
        return 0.5 * float(bg_w), 0.5 * float(bg_h)

    def _has_locked_imported_track(self) -> bool:
        overlay = self.project_state.imported_track
        return overlay.has_data and overlay.locked and bool(overlay.control_points_local_m)

    def _transform_imported_local_points(self, points: Sequence[Tuple[float, float]]) -> np.ndarray:
        overlay = self.project_state.imported_track
        if not points:
            return np.empty((0, 2), dtype=float)
        return transform_points(
            points,
            rotation_deg=overlay.rotation_deg,
            translation=(overlay.center_x_px, overlay.center_y_px),
            scale=self.project_state.background.px_per_m,
        )

    def _materialize_imported_control_points(self):
        overlay = self.project_state.imported_track
        if not overlay.has_data or not overlay.control_points_local_m:
            return
        self.project_state.control_points = array_to_points(self._transform_imported_local_points(overlay.control_points_local_m))

    def _transformed_imported_cones(self):
        overlay = self.project_state.imported_track
        if not overlay.has_data:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
        return (
            self._transform_imported_local_points(overlay.left_cones_local_m),
            self._transform_imported_local_points(overlay.right_cones_local_m),
        )

    def _capture_control_points_into_imported_track(self):
        overlay = self.project_state.imported_track
        if not overlay.has_data:
            return
        arr = points_to_array(self.project_state.control_points)
        if arr.size == 0:
            overlay.control_points_local_m = []
            return
        center = np.mean(arr, axis=0)
        overlay.center_x_px = float(center[0])
        overlay.center_y_px = float(center[1])
        shifted = arr - center
        if abs(overlay.rotation_deg) > 1e-9:
            theta = math.radians(-overlay.rotation_deg)
            rotation = np.array(
                [
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)],
                ],
                dtype=float,
            )
            shifted = shifted @ rotation.T
        local = shifted / max(self.project_state.background.px_per_m, 1e-6)
        overlay.control_points_local_m = array_to_points(local)

    def _track_transform_gizmo(self):
        if self.edit_target != "track" or not self._has_locked_imported_track() or not self.project_state.control_points:
            return None
        arr = points_to_array(self.project_state.control_points)
        center = np.mean(arr, axis=0)
        span_x = float(np.ptp(arr[:, 0])) if arr.shape[0] > 1 else 0.0
        span_y = float(np.ptp(arr[:, 1])) if arr.shape[0] > 1 else 0.0
        arm = min(220.0, max(70.0, 0.22 * max(span_x, span_y, 120.0)))
        rotate_radius = 1.45 * arm
        cx, cy = float(center[0]), float(center[1])
        return {
            "center": (cx, cy),
            "translate_x": (cx + arm, cy),
            "translate_y": (cx, cy - arm),
            "rotate": (cx, cy - rotate_radius),
        }

    def _toggle_imported_track_lock(self):
        overlay = self.project_state.imported_track
        if not overlay.has_data:
            return
        if overlay.locked:
            overlay.locked = False
            self.statusBar().showMessage("Unlocked track shape")
        else:
            self._capture_control_points_into_imported_track()
            overlay.locked = True
            self.statusBar().showMessage("Locked track shape")
        self._update_imported_track_controls()
        self._update_editor_controls()
        self._update_canvas()

    def _update_imported_track_controls(self):
        overlay = self.project_state.imported_track
        has_data = overlay.has_data
        self.clear_csv_button.setEnabled(has_data)
        self.toggle_shape_lock_button.setEnabled(has_data)
        if not has_data:
            self.imported_track_label.setText("Loaded CSV: none")
            self.imported_track_state_label.setText("Imported track: none")
            self.toggle_shape_lock_button.setText("Unlock Shape")
            self.overlay_hint_label.setText(
                "Load a CSV to create an editable centerline. Locked imported tracks can be translated or rotated with the on-canvas handles."
            )
            self._update_editor_controls()
            return

        source = overlay.source_path or "imported track"
        self.imported_track_label.setText(f"Loaded CSV: {source}")
        state = "shape locked" if overlay.locked else "shape unlocked"
        cones = "imported cones preserved" if overlay.preserve_cones else "generated cones active"
        self.imported_track_state_label.setText(f"Imported track: {state}, {cones}")
        self.toggle_shape_lock_button.setText("Unlock Shape" if overlay.locked else "Lock Shape")
        if overlay.locked:
            self.overlay_hint_label.setText(
                "Drag the on-canvas handles to translate or rotate the whole track. Imported cone positions remain fixed relative to the track while locked."
            )
        elif overlay.preserve_cones:
            self.overlay_hint_label.setText(
                "Shape editing is enabled. The imported cone positions will be replaced once the centerline or profiles change."
            )
        else:
            self.overlay_hint_label.setText(
                "Shape editing is enabled. Cone positions now follow the current centerline and width/spacing profiles."
            )
        self._update_editor_controls()

    def _confirm_import_overwrite(self) -> bool:
        if not (self.project_state.control_points or self.track_geometry.has_track or self.project_state.imported_track.has_data):
            return True
        result = QMessageBox.question(
            self,
            "Import CSV",
            "Importing a CSV will overwrite the current centerline track and discard cone overrides. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return result == QMessageBox.Yes

    def _load_track_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Track CSV", "", "CSV files (*.csv)")
        if not filename:
            return
        if not self._confirm_import_overwrite():
            return
        try:
            imported = load_track_csv(filename)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load Track CSV", str(exc))
            return

        center_x, center_y = self._overlay_center_default()
        self._clear_cone_overrides()
        self.track_width_function.set_constant(imported.track_width_m)
        self.cone_spacing_function.set_constant(imported.cone_spacing_m)
        self.track_width_entry.setText(f"{imported.track_width_m:.3f}")
        self.cone_spacing_entry.setText(f"{imported.cone_spacing_m:.3f}")
        self.project_state.imported_track = TrackOverlay(
            source_path=filename,
            left_cones_local_m=imported.left_cones_local_m,
            right_cones_local_m=imported.right_cones_local_m,
            control_points_local_m=imported.control_points_local_m,
            center_x_px=center_x,
            center_y_px=center_y,
            rotation_deg=0.0,
            locked=True,
            preserve_cones=True,
        )
        self.manual_mode_button.setChecked(True)
        self.edit_target = "track"
        self.track_button.setChecked(True)
        self._materialize_imported_control_points()
        self._update_imported_track_controls()
        self._update_editor_controls()
        self._rebuild_track()
        self.statusBar().showMessage(f"Loaded track CSV {filename}")

    def _clear_imported_track(self):
        self.project_state.imported_track = TrackOverlay()
        self._update_imported_track_controls()
        self._update_editor_controls()
        self._rebuild_track()
        self.statusBar().showMessage("Cleared imported track state")

    def _start_track_transform_drag(self, kind: str, point: Tuple[float, float]):
        overlay = self.project_state.imported_track
        if not self._has_locked_imported_track():
            return
        self.dragging_kind = kind
        self._transform_drag_start_point = np.asarray(point, dtype=float)
        self._transform_drag_start_center = np.array([overlay.center_x_px, overlay.center_y_px], dtype=float)
        self._transform_drag_start_rotation = float(overlay.rotation_deg)
        self._transform_drag_start_angle = math.atan2(point[1] - overlay.center_y_px, point[0] - overlay.center_x_px)

    def _apply_track_transform_drag(self, point: Tuple[float, float]):
        overlay = self.project_state.imported_track
        if not self._has_locked_imported_track() or self._transform_drag_start_center is None:
            return
        current = np.asarray(point, dtype=float)
        if self.dragging_kind == "track_translate_free" and self._transform_drag_start_point is not None:
            delta = current - self._transform_drag_start_point
            overlay.center_x_px = float(self._transform_drag_start_center[0] + delta[0])
            overlay.center_y_px = float(self._transform_drag_start_center[1] + delta[1])
        elif self.dragging_kind == "track_translate_x" and self._transform_drag_start_point is not None:
            delta_x = current[0] - self._transform_drag_start_point[0]
            overlay.center_x_px = float(self._transform_drag_start_center[0] + delta_x)
            overlay.center_y_px = float(self._transform_drag_start_center[1])
        elif self.dragging_kind == "track_translate_y" and self._transform_drag_start_point is not None:
            delta_y = current[1] - self._transform_drag_start_point[1]
            overlay.center_x_px = float(self._transform_drag_start_center[0])
            overlay.center_y_px = float(self._transform_drag_start_center[1] + delta_y)
        elif self.dragging_kind == "track_rotate" and self._transform_drag_start_rotation is not None and self._transform_drag_start_angle is not None:
            center = self._transform_drag_start_center
            current_angle = math.atan2(current[1] - center[1], current[0] - center[0])
            overlay.rotation_deg = float(self._transform_drag_start_rotation + math.degrees(current_angle - self._transform_drag_start_angle))
        self._rebuild_track()

    def _discard_preserved_imported_cones(self):
        overlay = self.project_state.imported_track
        if overlay.has_data and not overlay.locked and overlay.preserve_cones:
            overlay.preserve_cones = False
            self._update_imported_track_controls()

    def _show_help(self):
        QMessageBox.information(
            self,
            "TrackDraw Help",
            (
                "1. Choose a background image or metric grid and confirm the px/m scale.\n"
                "2. Define the allowed area polygon, then add, generate, or import centerline control points.\n"
                "3. Adjust the width and cone-spacing profiles, validate the layout, and export CSV when ready.\n\n"
                "Tips:\n"
                "- Middle mouse pans and the wheel zooms.\n"
                "- Metric grids use the configured grid spacing in meters.\n"
                "- Loading a CSV creates a centerline immediately and starts with the shape locked.\n"
                "- While locked, use the on-canvas handles to translate or rotate the whole imported track."
            ),
        )

    def _format_background_info(self) -> str:
        scale = f"Scale: {self.project_state.background.px_per_m:.3f} px/m"
        if self.project_state.background.kind == "grid":
            spacing = self.project_state.background.grid_spacing_m
            return f"{scale} | Grid: {spacing:.2f} m minor / {spacing * 5.0:.2f} m major"
        name = self.project_state.background.location_name or "location image"
        return f"{scale} | Background: {name}"

    def _editable_cone_array(self, side: str) -> np.ndarray:
        if side == "left":
            return np.asarray(self.track_geometry.left_cones, dtype=float).copy()
        return np.asarray(self.track_geometry.right_cones, dtype=float).copy()

    def _set_editable_cone_array(self, side: str, cones: np.ndarray):
        if side == "left":
            self.track_geometry.left_cones = np.asarray(cones, dtype=float)
        else:
            self.track_geometry.right_cones = np.asarray(cones, dtype=float)

    def _find_near_point(self, points: Sequence[Tuple[float, float]], target: Tuple[float, float], threshold: float = 14.0) -> Optional[int]:
        if not points:
            return None
        best_index = None
        best_dist = None
        tx, ty = target
        for index, (x, y) in enumerate(points):
            dist = (x - tx) ** 2 + (y - ty) ** 2
            if dist <= threshold ** 2 and (best_dist is None or dist < best_dist):
                best_index = index
                best_dist = dist
        return best_index

    def _find_near_cone(self, target: Tuple[float, float], threshold: float = 18.0) -> Optional[Tuple[str, int]]:
        tx, ty = target
        target_arr = np.array([tx, ty], dtype=float)
        best = None
        for side, cones in (("left", self.track_geometry.left_cones), ("right", self.track_geometry.right_cones)):
            if cones is None:
                continue
            arr = np.asarray(cones, dtype=float)
            if arr.size == 0:
                continue
            diff = arr - target_arr
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            index = int(np.argmin(dist_sq))
            if dist_sq[index] <= threshold ** 2 and (best is None or dist_sq[index] < best[2]):
                best = (side, index, dist_sq[index])
        if best is None:
            return None
        return best[0], best[1]

    def _confirm_discard_cone_overrides(self, message: str) -> bool:
        if not (self.project_state.left_cone_overrides or self.project_state.right_cone_overrides):
            return True
        result = QMessageBox.question(
            self,
            "Discard Cone Overrides",
            message + "\n\nYour manual cone edits will be discarded.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if result != QMessageBox.Yes:
            return False
        self._clear_cone_overrides()
        return True

    def _clear_cone_overrides(self):
        self.project_state.left_cone_overrides = []
        self.project_state.right_cone_overrides = []

    def _store_cone_overrides(self):
        self.project_state.left_cone_overrides = array_to_points(np.asarray(self.track_geometry.left_cones, dtype=float))
        self.project_state.right_cone_overrides = array_to_points(np.asarray(self.track_geometry.right_cones, dtype=float))

    def _rebuild_track(self):
        self._sync_project_state()
        overlay = self.project_state.imported_track
        if overlay.has_data and overlay.control_points_local_m:
            self._materialize_imported_control_points()
        self.track_geometry = build_track_geometry(
            control_points=self.project_state.control_points,
            px_per_m=self.project_state.background.px_per_m,
            n_points_midline=self.defaults["n_points_midline"],
            width_function=self.track_width_function,
            cone_spacing_function=self.cone_spacing_function,
        )
        if self.track_geometry.has_track and overlay.has_data and overlay.preserve_cones:
            left, right = self._transformed_imported_cones()
            set_cone_overrides(self.track_geometry, left, right)
        elif self.track_geometry.has_track and (self.project_state.left_cone_overrides or self.project_state.right_cone_overrides):
            set_cone_overrides(
                self.track_geometry,
                self.project_state.left_cone_overrides,
                self.project_state.right_cone_overrides,
            )
        self._revalidate()
        self._update_canvas()

    def _revalidate(self):
        self.validation_result = validate_track(
            self.project_state.allowed_area_points,
            self.track_geometry,
            self.project_state.background.px_per_m,
            self.project_state.rules,
        )
        self._update_validation_panel()
        self._update_stats()

    def _update_canvas(self):
        self.canvas.update_scene(
            allowed_area=self.project_state.allowed_area_points,
            control_points=self.project_state.control_points,
            centerline=self.track_geometry.centerline,
            left_boundary=self.track_geometry.left_boundary,
            right_boundary=self.track_geometry.right_boundary,
            left_cones=self.track_geometry.left_cones,
            right_cones=self.track_geometry.right_cones,
            edit_target=self.edit_target,
            transform_gizmo=self._track_transform_gizmo(),
        )

    def _update_validation_panel(self):
        self.validation_list.clear()
        if not self.validation_result.issues:
            self.validation_summary_label.setText("Validation: OK")
            item = QListWidgetItem("All enabled rules pass.")
            item.setForeground(QColor(28, 120, 58))
            self.validation_list.addItem(item)
            return

        self.validation_summary_label.setText(
            f"Validation: {self.validation_result.error_count} errors, {self.validation_result.warning_count} warnings"
        )
        for issue in self.validation_result.issues:
            item = QListWidgetItem(f"[{issue.severity.upper()}] {issue.summary}: {issue.detail}")
            item.setForeground(QColor(184, 34, 34) if issue.severity == "error" else QColor(184, 124, 0))
            self.validation_list.addItem(item)

    def _update_stats(self):
        if not self.track_geometry.has_track:
            self.track_length_label.setText("Track length: --")
            self.min_radius_label.setText("Min radius: --")
            self.cone_count_label.setText("Cones: --")
        else:
            left_count = len(self.track_geometry.left_cones) if self.track_geometry.left_cones is not None else 0
            right_count = len(self.track_geometry.right_cones) if self.track_geometry.right_cones is not None else 0
            self.track_length_label.setText(f"Track length: {self.track_geometry.track_length_m:.2f} m")
            self.min_radius_label.setText(f"Min radius: {self.track_geometry.min_radius_m:.2f} m")
            self.cone_count_label.setText(f"Cones: blue {left_count}, yellow {right_count}, total {left_count + right_count}")
        self.background_info_label.setText(self._format_background_info())

    @staticmethod
    def _read_float_entry(entry: QLineEdit, fallback: float, minimum: Optional[float] = 0.0) -> float:
        try:
            value = float(entry.text().strip())
        except ValueError:
            value = fallback
        if minimum is not None and value < minimum:
            value = minimum
        entry.setText(f"{value:.3f}")
        return value

    @staticmethod
    def _read_int_entry(entry: QLineEdit, fallback: int) -> int:
        try:
            value = int(entry.text().strip())
        except ValueError:
            value = fallback
        entry.setText(str(value))
        return value
