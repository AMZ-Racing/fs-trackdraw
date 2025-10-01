import os
import math
import csv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QSizePolicy,
    QToolButton,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap
import yaml
from utils_qt import (
    create_closed_spline,
    generate_oneside_boundary,
    generate_variable_offset_boundaries,
    sample_cones_variable,
    compute_curvature_profile,
)
from track_canvas_qt import TrackCanvas
from parameter_function import ParameterFunction
from function_editor_qt import FunctionEditorDialog
from min_curvature_calculation import optimize_raceline


class FSTrackDraw(QMainWindow):
    CURVATURE_ALERT_THRESHOLD = 1.0 / 4.5  # 1/m (diameter 9 m => radius 4.5 m)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrackDraw - PyQt")
        self.setMinimumSize(1024, 720)
        self.resize(1280, 860)
        
        # Global configuration file
        self.config_file = "config/track_config.yaml"
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
            self.track_width = config.get('track_width', 3.0)
            self.default_cone_distance = config.get('cone_distance', 3.5)
            self.min_boundary_backoff = config.get('min_boundary_backoff', 10.0)
            self.n_points_midline = config.get('n_points_midline', 300)
            self.location_name = config.get('standard_location', "empty")

        # Parameter functions over track progress
        self.track_width_function = ParameterFunction(self.track_width, name="Track Width")
        self.cone_spacing_function = ParameterFunction(self.default_cone_distance, name="Cone Spacing")

        # Auto spacing parameters (meters)
        self.auto_spacing_multiplier = 0.7
        self.auto_spacing_min_spacing = 1.0
        self.auto_spacing_max_spacing = 5.0

        # Randomization controls
        self.control_point_nudge_amount_m = 0.3
        self.cone_spacing_jitter_enabled = False
        self.cone_spacing_jitter_amount = 0.1
        self.cone_spacing_jitter_function = None
        self._cone_spacing_random_sample_index = 0
        self.centerline_warn_segments = []
        self.left_boundary_warn_segments = []
        self.right_boundary_warn_segments = []
        self.curvature_analysis_green_segments = []
        self.curvature_analysis_red_segments = []
        self.curvature_clearance_m = 0.5
        self.curvature_spacing_sequence = [1.0, 0.7, 0.2]

        # Load the location-specific details
        self.folderpath_location = "location_images/" + self.location_name
        self.filename_location_config = self.location_name + "_config.yaml"
        self.fpath_location_config = os.path.join(self.folderpath_location, self.filename_location_config)
        with open(self.fpath_location_config, 'r') as file:
            config = yaml.safe_load(file)
            self.px_per_m = config.get('px_per_m', 10.0)
            sat_img_file = config.get('sat_img_path', '')
            self.fpath_location_sat_img = os.path.join(self.folderpath_location, sat_img_file)
        
        # Editor state variables
        self.editor_mode = "control"  # "control" or "cone"
        self.mode = "add"  # Track submode: "add", "remove", "move"
        self.cone_mode = "move"  # Cone submode: "add", "move", "remove"
        self.selected_point_index = None
        self.selected_cone_side = None
        self.selected_cone_index = None
        self.dragging = False
        self.cone_dragging = False
        self.dragging_barrier = False
        self.barrier_mode = "add"  # Default mode is adding barrier points
        self.cone_edits_dirty = False

        self.track_editor_widgets = []
        self.cone_editor_widgets = []

        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # Create canvas (handles its own obstacle loading with proper scaling)
        self.canvas = TrackCanvas(self)
        self.canvas.setMinimumSize(500, 500)
        self.main_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.canvas, 1)
        
        # Right-side UI
        self.ui_frame = QWidget()
        self.ui_frame.setFixedWidth(250)
        self.ui_layout = QVBoxLayout(self.ui_frame)
        self.ui_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(self.ui_frame)
        
        # Editor mode selector
        selector_container = QWidget()
        selector_layout = QHBoxLayout(selector_container)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(6)

        self.editor_button_group = QButtonGroup(self)
        self.editor_button_group.setExclusive(True)

        self.track_editor_button = QPushButton("Track Editor")
        self.track_editor_button.setCheckable(True)
        self.track_editor_button.setChecked(True)
        self.track_editor_button.clicked.connect(lambda: self.set_editor_mode("control"))
        selector_layout.addWidget(self.track_editor_button, 1)
        self.editor_button_group.addButton(self.track_editor_button)

        self.cone_editor_button = QPushButton("Cone Editor")
        self.cone_editor_button.setCheckable(True)
        self.cone_editor_button.setChecked(False)
        self.cone_editor_button.clicked.connect(lambda: self.set_editor_mode("cone"))
        selector_layout.addWidget(self.cone_editor_button, 1)
        self.editor_button_group.addButton(self.cone_editor_button)

        self.ui_layout.addWidget(selector_container)

        # Mode label
        self.mode_label = QLabel("Track Mode: Add")
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.ui_layout.addWidget(self.mode_label)

        # Mode buttons
        self.add_button = QPushButton("Add Control Point")
        self.add_button.clicked.connect(self.activate_add_mode)
        self.ui_layout.addWidget(self.add_button)
        self.track_editor_widgets.append(self.add_button)

        self.remove_button = QPushButton("Remove Control Point")
        self.remove_button.clicked.connect(self.activate_remove_mode)
        self.ui_layout.addWidget(self.remove_button)
        self.track_editor_widgets.append(self.remove_button)

        self.move_button = QPushButton("Move Control Point")
        self.move_button.clicked.connect(self.activate_move_mode)
        self.ui_layout.addWidget(self.move_button)
        self.track_editor_widgets.append(self.move_button)

        # Cone mode buttons (disabled until cone editor active)
        cone_info_layout = QHBoxLayout()
        cone_info_layout.setContentsMargins(0, 0, 0, 0)
        cone_info_layout.setSpacing(4)

        self.add_cone_button = QPushButton("Add Cone")
        self.add_cone_button.clicked.connect(self.activate_cone_add_mode)
        cone_info_layout.addWidget(self.add_cone_button)

        self.cone_add_info_button = QToolButton()
        self.cone_add_info_button.setText("?")
        self.cone_add_info_button.setToolTip(
            "Cone Add mode: click an existing cone to insert a new cone halfway to the next cone in order."
        )
        self.cone_add_info_button.clicked.connect(self.show_add_cone_info)
        cone_info_layout.addWidget(self.cone_add_info_button)

        cone_add_container = QWidget()
        cone_add_container.setLayout(cone_info_layout)
        self.ui_layout.addWidget(cone_add_container)
        self.cone_editor_widgets.extend([cone_add_container, self.add_cone_button, self.cone_add_info_button])

        self.move_cone_button = QPushButton("Move Cone")
        self.move_cone_button.clicked.connect(self.activate_cone_move_mode)
        self.ui_layout.addWidget(self.move_cone_button)
        self.cone_editor_widgets.append(self.move_cone_button)

        self.remove_cone_button = QPushButton("Remove Cone")
        self.remove_cone_button.clicked.connect(self.activate_cone_remove_mode)
        self.ui_layout.addWidget(self.remove_cone_button)
        self.cone_editor_widgets.append(self.remove_cone_button)

        for widget in self.cone_editor_widgets:
            widget.setEnabled(False)

        self.ui_layout.addWidget(QLabel("Control point random nudge (m):"))
        self.control_point_nudge_entry = QLineEdit(f"{self.control_point_nudge_amount_m:.3f}")
        self.control_point_nudge_entry.setToolTip("Maximum absolute displacement applied per control point when randomizing")
        self.control_point_nudge_entry.editingFinished.connect(self.update_control_point_nudge_amount)
        self.ui_layout.addWidget(self.control_point_nudge_entry)
        self.track_editor_widgets.append(self.control_point_nudge_entry)

        self.randomize_control_points_button = QPushButton("Randomize Control Points")
        self.randomize_control_points_button.setToolTip("Apply random jitters to the current control points within the specified distance")
        self.randomize_control_points_button.clicked.connect(self.randomize_control_points)
        self.ui_layout.addWidget(self.randomize_control_points_button)
        self.track_editor_widgets.append(self.randomize_control_points_button)

        self.control_point_random_status_label = QLabel("Control point randomization: not applied")
        self.ui_layout.addWidget(self.control_point_random_status_label)

        # Swap boundaries button
        self.swap_button = QPushButton("Swap Boundaries")
        self.swap_button.clicked.connect(self.swap_boundaries)
        self.ui_layout.addWidget(self.swap_button)
        
        # Export button
        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.ui_layout.addWidget(self.export_button)
        
        # Cone spacing input
        self.ui_layout.addWidget(QLabel("Cone spacing (m):"))
        self.cone_spacing_entry = QLineEdit(str(self.default_cone_distance))
        self.cone_spacing_entry.returnPressed.connect(self.update_cone_spacing_constant)
        self.ui_layout.addWidget(self.cone_spacing_entry)
        self.track_editor_widgets.append(self.cone_spacing_entry)

        self.edit_cone_spacing_button = QPushButton("Edit Cone Spacing Function")
        self.edit_cone_spacing_button.clicked.connect(self.open_cone_spacing_editor)
        self.ui_layout.addWidget(self.edit_cone_spacing_button)
        self.track_editor_widgets.append(self.edit_cone_spacing_button)

        self.auto_spacing_checkbox = QCheckBox("Auto spacing from curvature")
        self.auto_spacing_checkbox.stateChanged.connect(self.toggle_auto_spacing)
        self.ui_layout.addWidget(self.auto_spacing_checkbox)
        self.track_editor_widgets.append(self.auto_spacing_checkbox)

        self.cone_spacing_random_checkbox = QCheckBox("Enable cone spacing randomness")
        self.cone_spacing_random_checkbox.stateChanged.connect(self.toggle_cone_spacing_randomness)
        self.cone_spacing_random_checkbox.setToolTip("Apply random variation to cone spacing values when enabled")
        self.ui_layout.addWidget(self.cone_spacing_random_checkbox)
        self.track_editor_widgets.append(self.cone_spacing_random_checkbox)

        self.ui_layout.addWidget(QLabel("Cone spacing randomness amount"))
        self.cone_spacing_random_amount_entry = QLineEdit(f"{self.cone_spacing_jitter_amount:.3f}")
        self.cone_spacing_random_amount_entry.setToolTip("Maximum fractional jitter (e.g. 0.10 = ±10%)")
        self.cone_spacing_random_amount_entry.editingFinished.connect(self.update_cone_spacing_jitter_amount)
        self.cone_spacing_random_amount_entry.setEnabled(False)
        self.ui_layout.addWidget(self.cone_spacing_random_amount_entry)
        self.track_editor_widgets.append(self.cone_spacing_random_amount_entry)

        self.randomize_cone_spacing_button = QPushButton("Randomize Cone Spacing")
        self.randomize_cone_spacing_button.setToolTip("Generate a new random cone spacing profile using the jitter amount")
        self.randomize_cone_spacing_button.clicked.connect(lambda: self.randomize_cone_spacing_profile())
        self.randomize_cone_spacing_button.setEnabled(False)
        self.ui_layout.addWidget(self.randomize_cone_spacing_button)
        self.track_editor_widgets.append(self.randomize_cone_spacing_button)

        self.cone_spacing_random_status_label = QLabel("Cone spacing randomness: disabled")
        self.ui_layout.addWidget(self.cone_spacing_random_status_label)
        self.track_editor_widgets.append(self.cone_spacing_random_status_label)

        self.check_curvature_button = QPushButton("Check Curvature")
        self.check_curvature_button.setToolTip(
            "Compute the minimum-curvature path and highlight sections tighter than 4.5 m radius"
        )
        self.check_curvature_button.clicked.connect(self.check_curvature)
        self.ui_layout.addWidget(self.check_curvature_button)

        self.curvature_check_status_label = QLabel("Curvature check: not run")
        self.ui_layout.addWidget(self.curvature_check_status_label)
        self.clear_curvature_analysis()

        self.ui_layout.addWidget(QLabel("Curvature clearance (m):"))
        self.curvature_clearance_entry = QLineEdit("0.50")
        self.curvature_clearance_entry.setToolTip("Minimum clearance passed to the curvature optimizer")
        self.curvature_clearance_entry.editingFinished.connect(self._sync_curvature_clearance)
        self.ui_layout.addWidget(self.curvature_clearance_entry)

        # Track width input
        self.ui_layout.addWidget(QLabel("Track width (m):"))
        self.track_width_entry = QLineEdit(str(self.track_width))
        self.track_width_entry.returnPressed.connect(self.update_track_width_constant)
        self.ui_layout.addWidget(self.track_width_entry)
        self.track_editor_widgets.append(self.track_width_entry)

        self.edit_track_width_button = QPushButton("Edit Track Width Function")
        self.edit_track_width_button.clicked.connect(self.open_track_width_editor)
        self.ui_layout.addWidget(self.edit_track_width_button)
        self.track_editor_widgets.append(self.edit_track_width_button)

        self._update_auto_spacing_tooltip()

        # Barrier Mode Label
        self.barrier_mode_label = QLabel("Barrier Mode: Add")
        self.barrier_mode_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.ui_layout.addWidget(self.barrier_mode_label)
        self.track_editor_widgets.append(self.barrier_mode_label)

        # Add barrier mode buttons
        self.add_barrier_button = QPushButton("Add Barrier Point")
        self.add_barrier_button.clicked.connect(self.activate_add_barrier_mode)
        self.ui_layout.addWidget(self.add_barrier_button)
        self.track_editor_widgets.append(self.add_barrier_button)

        self.move_barrier_button = QPushButton("Move Barrier Point")
        self.move_barrier_button.clicked.connect(self.activate_move_barrier_mode)
        self.ui_layout.addWidget(self.move_barrier_button)
        self.track_editor_widgets.append(self.move_barrier_button)

        self.remove_barrier_button = QPushButton("Remove Barrier Point")
        self.remove_barrier_button.clicked.connect(self.activate_remove_barrier_mode)
        self.ui_layout.addWidget(self.remove_barrier_button)
        self.track_editor_widgets.append(self.remove_barrier_button)

        # Swap barrier offset button
        self.swap_barrier_button = QPushButton("Swap Barrier Offset")
        self.swap_barrier_button.clicked.connect(self.swap_barrier_offset)
        self.ui_layout.addWidget(self.swap_barrier_button)
        self.track_editor_widgets.append(self.swap_barrier_button)

        # Backoff input
        self.ui_layout.addWidget(QLabel("Backoff (m):"))
        self.backoff_entry = QLineEdit(str(self.min_boundary_backoff))
        self.backoff_entry.returnPressed.connect(self.redraw)
        self.ui_layout.addWidget(self.backoff_entry)
        self.track_editor_widgets.append(self.backoff_entry)

        # Add label that explains to use right click for barrier points
        self.right_click_label = QLabel("Right Click: Barrier Points")
        self.ui_layout.addWidget(self.right_click_label)
        self.track_editor_widgets.append(self.right_click_label)
        # Add label that explains to use left click for track control points
        self.left_click_label = QLabel("Left Click: Track Control Points")
        self.ui_layout.addWidget(self.left_click_label)
        self.track_editor_widgets.append(self.left_click_label)

        # Statistics labels
        self.track_length_label = QLabel("Track Length: --")
        self.ui_layout.addWidget(self.track_length_label)
        
        self.min_radius_label = QLabel("Min Radius: --")
        self.ui_layout.addWidget(self.min_radius_label)
        
        self.cone_count_label = QLabel("Cones: \n Blue: -, Yellow: - \n Total: -")
        self.ui_layout.addWidget(self.cone_count_label)
        
        # Add stretch to push everything up
        self.ui_layout.addStretch(1)
        
        # Data for control points and track
        self.control_points = []  # List of QPointF
        self.barrier_polygon = []  # List of barrier points
        self.barrier_offset_polygon = []  # List of offset points
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.left_cones = None
        self.right_cones = None
        self.curvature_progress = None
        self.curvature_values = None
        self.curvature_spacing_profile = None
        self.centerline_warn_segments = []
        self.left_boundary_warn_segments = []
        self.right_boundary_warn_segments = []
        self.boundaries_swapped = False
        self.barrier_offset_swapped = False
        self.perform_swap = False
        self.perform_barrier_swap = False

        self.set_editor_mode("control")

        # Add logo at the bottom right
        self.logo_label = QLabel()
        self.logo_pixmap = QPixmap("TrackDraw_Logo.png")  # Load your logo image
        # Add the logo label to the layout
        self.ui_layout.addStretch(1)  # Push everything up
        self.ui_layout.addWidget(self.logo_label, 0, Qt.AlignCenter)
        # Handle window resize events
        self.main_widget.resizeEvent = self.on_resize
        # Initial logo setup
        self.update_logo_size()
        
        # Initialize auto spacing controls to manual mode
        self.toggle_auto_spacing(Qt.Unchecked)

        # Initialize
        self.redraw()

    def on_resize(self, event):
      self.update_logo_size()
      super().resizeEvent(event)  # Call parent's resize handler

    def update_logo_size(self):
      """Adjusts logo size when window is resized"""
      if hasattr(self, 'logo_pixmap') and self.logo_pixmap and not self.logo_pixmap.isNull():
          # Increased base size (adjust these values as needed)
          max_width = min(200, self.ui_frame.width() - 20)  # 20px padding
          scaled_pixmap = self.logo_pixmap.scaledToWidth(
              max_width, 
              Qt.SmoothTransformation
          )
          self.logo_label.setPixmap(scaled_pixmap)
          self.logo_label.setFixedSize(scaled_pixmap.size())  # Prevent layout shifting


    def update_cone_spacing_constant(self):
        try:
            value = float(self.cone_spacing_entry.text())
        except ValueError:
            value = self.default_cone_distance
        self.cone_spacing_function.set_constant(value)
        self.default_cone_distance = value
        self.cone_spacing_entry.setText(f"{value:.3f}")
        self.redraw()

    def update_track_width_constant(self):
        try:
            value = float(self.track_width_entry.text())
        except ValueError:
            value = self.track_width
        self.track_width_function.set_constant(value)
        self.track_width = value
        self.track_width_entry.setText(f"{value:.3f}")
        self.redraw()

    def open_cone_spacing_editor(self):
        dialog = FunctionEditorDialog(
            self,
            self.cone_spacing_function,
            "Cone Spacing Function",
            units="m",
            reference=self.get_curvature_reference(),
            auto_spacing_params={
                "multiplier": self.auto_spacing_multiplier,
                "min": self.auto_spacing_min_spacing,
                "max": self.auto_spacing_max_spacing,
            },
            auto_spacing_callback=self.set_auto_spacing_params,
        )
        if dialog.exec_():
            self.cone_spacing_entry.setText(f"{self.cone_spacing_function.evaluate(0.0):.3f}")
            self.redraw()

    def open_track_width_editor(self):
        dialog = FunctionEditorDialog(
            self,
            self.track_width_function,
            "Track Width Function",
            units="m",
            reference=self.get_curvature_reference(),
        )
        if dialog.exec_():
            self.track_width_entry.setText(f"{self.track_width_function.evaluate(0.0):.3f}")
            self.redraw()

    def toggle_auto_spacing(self, state):
        enabled = state == Qt.Checked
        self.cone_spacing_entry.setEnabled(not enabled)
        # Still allow opening the editor so curvature parameters remain adjustable
        self.edit_cone_spacing_button.setEnabled(True)
        if enabled:
            if self.curvature_spacing_profile is None:
                self.cone_spacing_entry.setToolTip("Auto spacing requires a computed track curvature")
            else:
                self.cone_spacing_entry.setToolTip(self._auto_spacing_formula_text())
        else:
            self.cone_spacing_entry.setToolTip("")
        # Trigger redraw so cone placement updates with new mode
        self.redraw()

    def update_control_point_nudge_amount(self):
        self.control_point_nudge_amount_m = self._parse_positive_line_edit(
            self.control_point_nudge_entry,
            self.control_point_nudge_amount_m,
            minimum=0.0,
        )

    def randomize_control_points(self):
        if not self.control_points:
            return
        amount_m = self.control_point_nudge_amount_m
        if amount_m <= 0:
            return
        amount_px = amount_m * self.px_per_m
        rng = np.random.default_rng()
        nudged = []
        for pt in self.control_points:
            dx, dy = rng.uniform(-amount_px, amount_px, size=2)
            nudged.append(QPointF(pt.x() + dx, pt.y() + dy))
        self.control_points = nudged
        self.control_point_random_status_label.setText(
            f"Control point randomization applied: ±{amount_m:.3f} m"
        )
        self.redraw()

    def _auto_spacing_formula_text(self) -> str:
        return (
            "spacing = clamp({mult:.3f} / κ, {min:.3f}, {max:.3f})\n"
            "κ is curvature (1/m); spacing is in meters."
        ).format(
            mult=self.auto_spacing_multiplier,
            min=self.auto_spacing_min_spacing,
            max=self.auto_spacing_max_spacing,
        )

    def _update_auto_spacing_tooltip(self):
        tooltip = (
            "Auto spacing from curvature uses the latest parameters set in the editor:\n"
            f"{self._auto_spacing_formula_text()}"
        )
        self.auto_spacing_checkbox.setToolTip(tooltip)

    def set_auto_spacing_params(self, multiplier: float, min_spacing: float, max_spacing: float):
        if multiplier <= 0:
            multiplier = self.auto_spacing_multiplier
        if min_spacing <= 0:
            min_spacing = self.auto_spacing_min_spacing
        if max_spacing < min_spacing:
            max_spacing = min_spacing

        self.auto_spacing_multiplier = float(multiplier)
        self.auto_spacing_min_spacing = float(min_spacing)
        self.auto_spacing_max_spacing = float(max_spacing)

        self._update_auto_spacing_tooltip()

        if self.auto_spacing_checkbox.isChecked():
            self.redraw()

    def update_cone_spacing_jitter_amount(self):
        self.cone_spacing_jitter_amount = self._parse_positive_line_edit(
            self.cone_spacing_random_amount_entry,
            self.cone_spacing_jitter_amount,
            minimum=0.0,
        )
        if self.cone_spacing_jitter_enabled:
            self.randomize_cone_spacing_profile()

    def toggle_cone_spacing_randomness(self, state):
        enabled = state == Qt.Checked
        self.cone_spacing_jitter_enabled = enabled
        self.cone_spacing_random_amount_entry.setEnabled(enabled)
        self.randomize_cone_spacing_button.setEnabled(enabled)
        if enabled:
            self.cone_spacing_random_status_label.setText("Cone spacing randomness: active")
            self.randomize_cone_spacing_profile()
        else:
            self.cone_spacing_jitter_function = None
            self._cone_spacing_random_sample_index = 0
            self.cone_spacing_random_status_label.setText("Cone spacing randomness: disabled")
            self.redraw()

    def randomize_cone_spacing_profile(self, redraw: bool = True):
        if not self.cone_spacing_jitter_enabled:
            return
        amount = max(0.0, float(self.cone_spacing_jitter_amount))
        # Clamp to avoid zero or negative factors
        amount = min(amount, 0.95)
        if not np.isclose(amount, self.cone_spacing_jitter_amount):
            self.cone_spacing_jitter_amount = amount
            self.cone_spacing_random_amount_entry.setText(f"{self.cone_spacing_jitter_amount:.3f}")

        num_points = max(8, self.n_points_midline // 6)
        progress = np.linspace(0.0, 1.0, num_points, endpoint=False)
        rng = np.random.default_rng()
        lower = max(0.05, 1.0 - amount)
        upper = 1.0 + amount
        factors = rng.uniform(lower, upper, size=progress.shape)
        control_points = list(zip(progress, factors))
        control_points.append((1.0, factors[0]))
        jitter_function = ParameterFunction(1.0, name="Cone Spacing Jitter")
        jitter_function.set_control_points(control_points)
        self.cone_spacing_jitter_function = jitter_function
        self._cone_spacing_random_sample_index += 1
        self.cone_spacing_random_status_label.setText(
            "Cone spacing randomness sample #{idx} (±{amt:.3f})".format(
                idx=self._cone_spacing_random_sample_index,
                amt=self.cone_spacing_jitter_amount,
            )
        )
        if redraw:
            self.redraw()

    def _compute_spacing_profile(self, curvature_values):
        curvature = np.asarray(curvature_values, dtype=float)
        if curvature.size == 0:
            return np.array([], dtype=float)
        denom = np.maximum(curvature, 1e-6)
        spacing = self.auto_spacing_multiplier / denom
        spacing = np.clip(spacing, self.auto_spacing_min_spacing, self.auto_spacing_max_spacing)
        return spacing

    def get_spacing_sampler(self):
        if self.auto_spacing_checkbox.isChecked() and self.curvature_spacing_profile is not None:
            base_sampler = self.curvature_spacing_sampler
        else:
            base_sampler = self.cone_spacing_function.evaluate_array

        if self.cone_spacing_jitter_enabled and self.cone_spacing_jitter_function is not None:
            jitter_function = self.cone_spacing_jitter_function

            def jittered_sampler(progress_array):
                base_values = np.asarray(base_sampler(progress_array), dtype=float)
                factors = np.asarray(jitter_function.evaluate_array(progress_array), dtype=float)
                result = base_values * factors
                if np.any(~np.isfinite(result)):
                    result = base_values
                return result

            return jittered_sampler

        return base_sampler

    def curvature_spacing_sampler(self, progress_array):
        if self.curvature_progress is None or self.curvature_spacing_profile is None:
            progress_array = np.asarray(progress_array, dtype=float)
            return np.full(progress_array.shape, self.default_cone_distance, dtype=float)
        progress = np.asarray(progress_array, dtype=float)
        wrapped = np.mod(progress, 1.0)
        xp = np.asarray(self.curvature_progress, dtype=float)
        fp = np.asarray(self.curvature_spacing_profile, dtype=float)
        if xp.size == 0 or fp.size == 0:
            return np.full_like(progress, self.default_cone_distance, dtype=float)
        xp = np.concatenate((xp, [1.0]))
        fp = np.concatenate((fp, [fp[0]]))
        return np.interp(wrapped, xp, fp)

    def get_curvature_reference(self):
        if self.curvature_progress is None or self.curvature_values is None:
            return None
        return {
            "progress": self.curvature_progress,
            "values": self.curvature_values,
            "label": "Curvature κ (1/m)",
        }
    
    def update_mode_label(self):
        if self.editor_mode == "control":
            mode_names = {"add": "Add", "remove": "Remove", "move": "Move"}
            label = mode_names.get(self.mode, self.mode.title())
            self.mode_label.setText(f"Track Mode: {label}")
        else:
            mode_names = {"add": "Add", "remove": "Remove", "move": "Move"}
            label = mode_names.get(self.cone_mode, self.cone_mode.title())
            self.mode_label.setText(f"Cone Mode: {label}")

    def activate_add_mode(self):
        if self.editor_mode != "control":
            self.set_editor_mode("control")
        self.mode = "add"
        self.update_mode_label()
        
    def activate_remove_mode(self):
        if self.editor_mode != "control":
            self.set_editor_mode("control")
        self.mode = "remove"
        self.update_mode_label()
        
    def activate_move_mode(self):
        if self.editor_mode != "control":
            self.set_editor_mode("control")
        self.mode = "move"
        self.update_mode_label()

    def activate_cone_add_mode(self):
        if self.editor_mode != "cone":
            self.set_editor_mode("cone")
        self.cone_mode = "add"
        self.update_mode_label()

    def activate_cone_move_mode(self):
        if self.editor_mode != "cone":
            self.set_editor_mode("cone")
        self.cone_mode = "move"
        self.update_mode_label()

    def activate_cone_remove_mode(self):
        if self.editor_mode != "cone":
            self.set_editor_mode("cone")
        self.cone_mode = "remove"
        self.update_mode_label()

    def set_editor_mode(self, mode: str):
        mode = "cone" if mode == "cone" else "control"
        if self.editor_mode == "cone" and mode == "control":
            message = (
                "Switching back to the Track Editor will regenerate boundaries and cones,"
                " discarding cone edits made in Cone Editor mode."
            )
            if self.cone_edits_dirty:
                result = QMessageBox.question(
                    self,
                    "Leave Cone Editor",
                    message + "\n\nProceed?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if result != QMessageBox.Yes:
                    self.cone_editor_button.setChecked(True)
                    self.track_editor_button.setChecked(False)
                    return
            else:
                QMessageBox.information(self, "Leave Cone Editor", message)
            self.cone_edits_dirty = False

        self.editor_mode = mode

        self.track_editor_button.setChecked(mode == "control")
        self.cone_editor_button.setChecked(mode == "cone")

        for widget in self.track_editor_widgets:
            if mode == "cone":
                widget._track_prev_enabled = widget.isEnabled()
                widget.setEnabled(False)
            else:
                prev_enabled = getattr(widget, "_track_prev_enabled", widget.isEnabled())
                widget.setEnabled(prev_enabled)

        for widget in self.cone_editor_widgets:
            widget.setEnabled(mode == "cone")

        if mode == "cone":
            if self.cone_mode not in {"add", "move", "remove"}:
                self.cone_mode = "move"
            self.dragging = False
            self.selected_point_index = None
        else:
            if self.mode not in {"add", "remove", "move"}:
                self.mode = "add"
            self.cone_dragging = False
            self.selected_cone_side = None
            self.selected_cone_index = None

        self.update_mode_label()
        self.canvas.set_editor_mode(mode)
        self._update_canvas()

    def show_add_cone_info(self):
        QMessageBox.information(
            self,
            "Adding Cones",
            (
                "In Add Cone mode, click an existing cone. A new cone will be inserted"
                " midway between the selected cone and the next cone in that"
                " direction, preserving ordering."
            ),
        )

    def activate_add_barrier_mode(self):
        self.barrier_mode = "add"
        self.barrier_mode_label.setText("Barrier Mode: Add")

    def activate_move_barrier_mode(self):
        self.barrier_mode = "move"
        self.barrier_mode_label.setText("Barrier Mode: Move")

    def activate_remove_barrier_mode(self):
        self.barrier_mode = "remove"
        self.barrier_mode_label.setText("Barrier Mode: Remove")
        
    def swap_boundaries(self):
        self.boundaries_swapped = not self.boundaries_swapped
        if (
            self.editor_mode == "cone"
            and self.left_cones is not None
            and self.right_cones is not None
        ):
            self.left_cones, self.right_cones = self.right_cones, self.left_cones
            if self.left_boundary is not None and self.right_boundary is not None:
                self.left_boundary, self.right_boundary = self.right_boundary, self.left_boundary
            self._after_cone_edit(update_counts=True)
        else:
            self.perform_swap = True
            self.redraw()

    def swap_barrier_offset(self):
        self.barrier_offset_swapped = not self.barrier_offset_swapped
        self.perform_barrier_swap = True
        self.redraw()
    
    def handle_canvas_rightclick(self, pos):
        x, y = pos.x(), pos.y()
        if self.barrier_mode == "add":
            self.barrier_polygon.insert(0, QPointF(x, y))
            self.redraw()
        elif self.barrier_mode == "remove":
            idx = self.find_near_barrier_point(x, y)
            if idx is not None:
                del self.barrier_polygon[idx]
                self.redraw()
        elif self.barrier_mode == "move":
            idx = self.find_near_barrier_point(x, y, threshold=20)
            if idx is not None:
                self.selected_point_index = idx
                self.dragging_barrier = True
        
    def handle_canvas_click(self, pos):
        x, y = pos.x(), pos.y()
        if self.editor_mode == "cone":
            self._handle_cone_click(x, y)
            return

        if self.mode == "add":
            self.control_points.insert(0, QPointF(x, y))
            self.redraw()
        elif self.mode == "remove":
            idx = self.find_near_control_point(x, y)
            if idx is not None:
                del self.control_points[idx]
                self.redraw()
        elif self.mode == "move":
            idx = self.find_near_control_point(x, y, threshold=20)
            if idx is not None:
                self.selected_point_index = idx
                self.dragging = True

    def handle_canvas_drag(self, pos):
        if (
            self.editor_mode == "control"
            and self.dragging
            and self.selected_point_index is not None
            and self.mode == "move"
        ):
            self.control_points[self.selected_point_index] = QPointF(pos.x(), pos.y())
            self.redraw()
        elif (
            self.editor_mode == "cone"
            and self.cone_dragging
            and self.selected_cone_side is not None
            and self.selected_cone_index is not None
        ):
            self._handle_cone_drag(pos.x(), pos.y())
        elif self.dragging_barrier and self.selected_point_index is not None and self.barrier_mode == "move":
            self.barrier_polygon[self.selected_point_index] = QPointF(pos.x(), pos.y())
            self.redraw()
            
    def handle_canvas_release(self, pos):
        self.dragging = False
        self.dragging_barrier = False
        self.selected_point_index = None
        if self.cone_dragging:
            self._handle_cone_release()

    def find_near_control_point(self, x, y, threshold=10):
        for i, pt in enumerate(self.control_points):
            if (pt.x() - x) ** 2 + (pt.y() - y) ** 2 < threshold ** 2:
                return i
        return None
    
    def find_near_cone(self, x, y, threshold=15):
        best = None
        threshold_sq = threshold ** 2
        target = np.array([x, y], dtype=float)
        for side, cones in (("left", self.left_cones), ("right", self.right_cones)):
            arr = self._ensure_cone_array(cones)
            if arr.size == 0:
                continue
            diffs = arr - target
            dists_sq = np.einsum("ij,ij->i", diffs, diffs)
            idx = int(np.argmin(dists_sq))
            dist_sq = float(dists_sq[idx])
            if dist_sq <= threshold_sq and (best is None or dist_sq < best[2]):
                best = (side, idx, dist_sq)
        return best

    def find_near_barrier_point(self, x, y, threshold=10):
        if self.barrier_polygon is not None:
            for i, pt in enumerate(self.barrier_polygon):
                if (pt.x() - x) ** 2 + (pt.y() - y) ** 2 < threshold ** 2:
                    return i
        return None
        
    def _handle_cone_click(self, x: float, y: float):
        selection = self.find_near_cone(x, y, threshold=20)
        if selection is None:
            return
        side, idx, _ = selection

        if self.cone_mode == "add":
            arr = self._ensure_cone_array(self.left_cones if side == "left" else self.right_cones)
            if arr.shape[0] < 2:
                return
            next_idx = (idx + 1) % arr.shape[0]
            midpoint = 0.5 * (arr[idx] + arr[next_idx])
            arr = np.insert(arr, next_idx, midpoint, axis=0)
            if side == "left":
                self.left_cones = arr
            else:
                self.right_cones = arr
            self.clear_curvature_analysis("cones updated")
            self._after_cone_edit(update_counts=True)
        elif self.cone_mode == "remove":
            arr = self._ensure_cone_array(self.left_cones if side == "left" else self.right_cones)
            if arr.shape[0] == 0:
                return
            arr = np.delete(arr, idx, axis=0)
            if side == "left":
                self.left_cones = arr
            else:
                self.right_cones = arr
            self.clear_curvature_analysis("cones updated")
            self._after_cone_edit(update_counts=True)
        else:  # move mode
            self.selected_cone_side = side
            self.selected_cone_index = idx
            self.cone_dragging = True
            self.clear_curvature_analysis("cones updated")

    def _handle_cone_drag(self, x: float, y: float):
        if not self.cone_dragging or self.selected_cone_side is None or self.selected_cone_index is None:
            return
        side = self.selected_cone_side
        arr = self._ensure_cone_array(self.left_cones if side == "left" else self.right_cones)
        if arr.shape[0] == 0 or self.selected_cone_index >= arr.shape[0]:
            return
        arr[self.selected_cone_index] = np.array([x, y], dtype=float)
        if side == "left":
            self.left_cones = arr
        else:
            self.right_cones = arr
        self._after_cone_edit(update_counts=False, clear_curvature=False, mark_dirty=False)
        self.cone_edits_dirty = True

    def _handle_cone_release(self):
        self.cone_dragging = False
        self.selected_cone_side = None
        self.selected_cone_index = None
        self._after_cone_edit(update_counts=False, clear_curvature=False, mark_dirty=False)

    def _after_cone_edit(
        self,
        update_counts: bool = True,
        clear_curvature: bool = True,
        mark_dirty: bool = True,
    ):
        self.left_cones = self._ensure_cone_array(self.left_cones)
        self.right_cones = self._ensure_cone_array(self.right_cones)
        self._update_canvas()
        if update_counts:
            self._update_cone_count_label()
        if clear_curvature and (
            self.curvature_analysis_green_segments or self.curvature_analysis_red_segments
        ):
            self.clear_curvature_analysis("cones updated")
        if mark_dirty:
            self.cone_edits_dirty = True

    def _update_cone_count_label(self):
        if self.left_cones is None or self.right_cones is None:
            self.cone_count_label.setText("Cones: \n Blue: -, Yellow: - \n Total: -")
            return
        blue, yellow, total = self.count_cones()
        self.cone_count_label.setText(
            f"Cones: \n Blue: {blue}, Yellow: {yellow} \n Total: {total}"
        )

    @staticmethod
    def _ensure_cone_array(values):
        if values is None:
            return np.empty((0, 2), dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        return arr.reshape(-1, 2)

    def export_csv(self):
        """Export the cone positions as CSV in meters."""
        if (
            self.centerline is None
            or self.left_boundary is None
            or self.right_boundary is None
            or self.left_cones is None
            or self.right_cones is None
        ):
            QMessageBox.critical(self, "Export Error", "No track defined yet!")
            return
            
        left_cones = np.asarray(self.left_cones, dtype=float)
        right_cones = np.asarray(self.right_cones, dtype=float)

        # Define new coordinate system
        origin = np.array([self.centerline[0].x(), self.centerline[0].y()])
        if len(self.centerline) < 2:
            QMessageBox.critical(self, "Export Error", "Not enough centerline points!")
            return
            
        tangent = np.array([self.centerline[1].x(), self.centerline[1].y()]) - origin
        theta = math.atan2(tangent[1], tangent[0])
        
        # Rotation matrix for -theta
        R = np.array([[math.cos(-theta), -math.sin(-theta)],
                      [math.sin(-theta),  math.cos(-theta)]])
                      
        def transform(pt):
            local = np.array(pt) - origin
            local_rot = R.dot(local)
            return local_rot / self.px_per_m  # convert pixels to meters
            
        left_cones_m = [transform(pt) for pt in left_cones]
        right_cones_m = [transform(pt) for pt in right_cones]
        
        left_tag = "blue"
        right_tag = "yellow"
            
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not filename:
            return
            
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["tag", "x", "y"])
            for pt in right_cones_m:
                writer.writerow([right_tag, pt[0], pt[1]])
            for pt in left_cones_m:
                writer.writerow([left_tag, pt[0], pt[1]])
                
        QMessageBox.information(self, "Export", f"Track exported to {filename}")

    def redraw(self):
        if self.perform_swap:
            # reverse the order of the control points
            self.control_points.reverse()
            self.perform_swap = False

        if self.perform_barrier_swap:
            # reverse the order of the barrier points
            self.barrier_polygon.reverse()
            self.perform_barrier_swap = False
        
        # Update track parameters
        try:
            backoff_val = float(self.backoff_entry.text())
        except ValueError:
            backoff_val = self.min_boundary_backoff
        self.min_boundary_backoff = backoff_val

        if self.curvature_analysis_green_segments or self.curvature_analysis_red_segments:
            self.clear_curvature_analysis("track updated")

        # If at least 3 control points, compute and draw track
        track_ready = False
        self.centerline_warn_segments = []
        self.left_boundary_warn_segments = []
        self.right_boundary_warn_segments = []
        if len(self.control_points) > 3:
            pts = np.array([[p.x(), p.y()] for p in self.control_points])
            centerline_np = create_closed_spline(pts, num_points=self.n_points_midline)
            self.centerline = [QPointF(p[0], p[1]) for p in centerline_np]

            progress, curvature = compute_curvature_profile(centerline_np, self.px_per_m)
            if progress.size:
                spacing_profile = self._compute_spacing_profile(curvature)
                self.curvature_progress = progress
                self.curvature_values = curvature
                self.curvature_spacing_profile = spacing_profile
            else:
                self.curvature_progress = None
                self.curvature_values = None
                self.curvature_spacing_profile = None

            try:
                boundaries = generate_variable_offset_boundaries(
                    centerline_np,
                    self.track_width_function.evaluate_array,
                    self.px_per_m,
                )
            except Exception as exc:
                boundaries = (None, None)
                print("Failed to build offset boundaries:", exc)

            if boundaries[0] is not None and boundaries[1] is not None:
                left_boundary, right_boundary = boundaries
                self.left_boundary = [QPointF(p[0], p[1]) for p in left_boundary]
                self.right_boundary = [QPointF(p[0], p[1]) for p in right_boundary]

                spacing_sampler = self.get_spacing_sampler()
                try:
                    self.left_cones = sample_cones_variable(
                        left_boundary, centerline_np, spacing_sampler, self.px_per_m
                    )
                except Exception as exc:
                    print("Failed to sample left cones:", exc)
                    self.left_cones = None
                try:
                    self.right_cones = sample_cones_variable(
                        right_boundary, centerline_np, spacing_sampler, self.px_per_m
                    )
                except Exception as exc:
                    print("Failed to sample right cones:", exc)
                    self.right_cones = None

                track_length = self.calculate_track_length()
                min_radius = self.calculate_min_radius()
                blue_cones, yellow_cones, total_cones = self.count_cones()

                self.track_length_label.setText(f"Track Length: {track_length:.2f} m")
                self.min_radius_label.setText(f"Min Radius: {min_radius:.2f} m")
                self.cone_count_label.setText(
                    f"Cones: \n Blue: {blue_cones}, Yellow: {yellow_cones} \n Total: {total_cones}"
                )
                self.cone_edits_dirty = False
                track_ready = True
            else:
                self.left_boundary = None
                self.right_boundary = None
                
        if not track_ready:
            self.centerline = None
            self.left_boundary = None
            self.right_boundary = None
            self.left_cones = None
            self.right_cones = None
            self.curvature_progress = None
            self.curvature_values = None
            self.curvature_spacing_profile = None
            self.track_length_label.setText("Track Length: --")
            self.min_radius_label.setText("Min Radius: --")
            self.cone_count_label.setText("Cones: \n Blue: -, Yellow: - \n Total: -")
            self.cone_edits_dirty = False

        if len(self.barrier_polygon) > 2:
            # Convert barrier polygon to numpy array for spline calculation
            pts = np.array([[p.x(), p.y()] for p in self.barrier_polygon])
            self.barrier_offset_polygon = generate_oneside_boundary(pts, self.min_boundary_backoff, self.px_per_m)
            if self.barrier_offset_polygon is not None:
                self.barrier_offset_polygon = [QPointF(p[0], p[1]) for p in self.barrier_offset_polygon]

        self._update_canvas()
            
    def _convert_to_qpoints(self, points):
        if points is None:
            return None
        converted = []
        for pt in points:
            if isinstance(pt, QPointF):
                converted.append(pt)
            else:
                converted.append(QPointF(float(pt[0]), float(pt[1])))
        return converted

    def _convert_segments_to_qpoints(self, segments):
        converted = []
        for p0, p1 in segments:
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            converted.append((QPointF(x0, y0), QPointF(x1, y1)))
        return converted

    def clear_curvature_analysis(self, reason: str = None):
        self.curvature_analysis_green_segments = []
        self.curvature_analysis_red_segments = []
        if reason:
            self.curvature_check_status_label.setText(f"Curvature check: {reason}")
        else:
            self.curvature_check_status_label.setText("Curvature check: not run")

    def _sync_curvature_clearance(self):
        self.curvature_clearance_m = self._parse_positive_line_edit(
            self.curvature_clearance_entry,
            self.curvature_clearance_m,
            minimum=0.05,
        )

    def _update_canvas(self):
        self.canvas.update_drawing(
            self.control_points,
            self.centerline,
            self.left_boundary,
            self.right_boundary,
            self.boundaries_swapped,
            self.barrier_polygon,
            self.barrier_offset_polygon,
            self._convert_to_qpoints(self.left_cones),
            self._convert_to_qpoints(self.right_cones),
            self.centerline_warn_segments,
            self.left_boundary_warn_segments,
            self.right_boundary_warn_segments,
            self._convert_segments_to_qpoints(self.curvature_analysis_green_segments),
            self._convert_segments_to_qpoints(self.curvature_analysis_red_segments),
            editor_mode=self.editor_mode,
        )

    def _build_curvature_analysis_segments(self, path_px: np.ndarray, curvature: np.ndarray, threshold: float):
        if path_px is None or curvature is None:
            return [], []
        points = np.asarray(path_px, dtype=float)
        curv = np.asarray(curvature, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return [], []
        if curv.shape[0] != points.shape[0]:
            count = min(points.shape[0], curv.shape[0])
            points = points[:count]
            curv = curv[:count]
        if np.allclose(points[0], points[-1]):
            points = points[:-1]
            curv = curv[: points.shape[0]]
        count = points.shape[0]
        if count < 2:
            return [], []
        threshold = abs(threshold)
        segments_green = []
        segments_red = []
        for idx in range(count):
            nxt = (idx + 1) % count
            p0 = points[idx]
            p1 = points[nxt]
            curv_ok = (abs(curv[idx]) <= threshold) and (abs(curv[nxt]) <= threshold)
            if curv_ok:
                segments_green.append((p0, p1))
            else:
                segments_red.append((p0, p1))
        return segments_green, segments_red

    def check_curvature(self):
        if self.left_cones is None or self.right_cones is None:
            QMessageBox.warning(self, "Curvature Check", "Generate the cones before checking curvature.")
            return
        left_points = np.asarray(self.left_cones, dtype=float)
        right_points = np.asarray(self.right_cones, dtype=float)
        if left_points.ndim != 2 or right_points.ndim != 2 or left_points.shape[1] != 2 or right_points.shape[1] != 2:
            QMessageBox.warning(self, "Curvature Check", "Cone arrays are malformed; cannot evaluate curvature.")
            return
        if left_points.shape[0] < 3 or right_points.shape[0] < 3:
            QMessageBox.warning(self, "Curvature Check", "Not enough cones to evaluate curvature.")
            return
        scale = float(self.px_per_m)
        left_m = left_points / scale
        right_m = right_points / scale
        self._sync_curvature_clearance()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.curvature_check_status_label.setText("Curvature check: computing…")
        try:
            x_opt, y_opt, _, _, curvature, _ = optimize_raceline(
                left_m,
                right_m,
                spacing_list=self.curvature_spacing_sequence,
                dist_weight=1e-5,
                wall_clearance=self.curvature_clearance_m,
                use_sparse=True,
                use_kdtree=True,
                closed_path=True,
                debug=False,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.clear_curvature_analysis("failed")
            QMessageBox.critical(self, "Curvature Check", f"Failed to compute minimum-curvature path:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        path_px = np.column_stack((x_opt, y_opt)) * scale
        green_segments, red_segments = self._build_curvature_analysis_segments(
            path_px, curvature, self.CURVATURE_ALERT_THRESHOLD
        )
        self.curvature_analysis_green_segments = green_segments
        self.curvature_analysis_red_segments = red_segments
        if red_segments:
            self.curvature_check_status_label.setText(
                f"Curvature check: {len(red_segments)} segments below 4.5 m radius"
            )
        else:
            self.curvature_check_status_label.setText(
                "Curvature check: OK (radius ≥ 4.5 m)"
            )
        self._update_canvas()

    def _parse_positive_line_edit(self, entry, current_value, minimum: float = 0.0):
        text = entry.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = current_value
        if value < minimum:
            value = minimum
        entry.setText(f"{value:.3f}")
        return value

    def calculate_track_length(self):
        """Calculate the total length of the track (centerline B-spline)."""
        if len(self.centerline) < 2:
            return 0
        total_length = 0
        for i in range(1, len(self.centerline)):
            x1, y1 = self.centerline[i - 1].x(), self.centerline[i - 1].y()
            x2, y2 = self.centerline[i].x(), self.centerline[i].y()
            total_length += math.hypot(x2 - x1, y2 - y1)
        return total_length / self.px_per_m  # Convert from pixels to meters
        
    def calculate_min_radius(self):
        """Calculate the minimum radius of curvature of the centerline B-spline."""
        if self.curvature_values is None or len(self.curvature_values) == 0:
            return float('inf')
        with np.errstate(divide='ignore'):
            radii = np.where(self.curvature_values > 1e-9, 1.0 / self.curvature_values, np.inf)
        return float(np.min(radii))
        
    def count_cones(self):
        """Count the number of blue, yellow, and total cones."""
        if self.left_cones is None or self.right_cones is None:
            return 0, 0, 0
        blue = len(self.left_cones)
        yellow = len(self.right_cones)
        return blue, yellow, blue + yellow
