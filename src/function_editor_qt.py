from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QDialog,
    QButtonGroup,
    QRadioButton,
    QFormLayout,
    QLineEdit,
)


class FunctionCanvas(QWidget):
    """Interactive canvas to edit progress-based control points."""

    def __init__(self, parent=None, units: str = ""):
        super().__init__(parent)
        self.units = units
        self.points: List[Tuple[float, float]] = [(0.0, 0.0), (1.0, 0.0)]
        self.mode = "add"
        self.drag_index = None
        self.setMinimumSize(640, 320)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def set_points(self, points: List[Tuple[float, float]]) -> None:
        if not points:
            points = [(0.0, 0.0), (1.0, 0.0)]
        # ensure sorted and unique progress
        pts = sorted(points, key=lambda item: item[0])
        filtered: List[Tuple[float, float]] = []
        for p, v in pts:
            if filtered and abs(p - filtered[-1][0]) < 1e-6:
                filtered[-1] = (p, 0.5 * (filtered[-1][1] + v))
            else:
                filtered.append((p, v))
        self.points = filtered
        self.update()

    def get_points(self) -> List[Tuple[float, float]]:
        return list(self.points)

    # Coordinate helpers --------------------------------------------------
    def _content_rect(self):
        margin = 40
        return (
            margin,
            margin,
            max(10, self.width() - 2 * margin),
            max(10, self.height() - 2 * margin),
        )

    def _value_bounds(self):
        values = [v for _, v in self.points]
        if not values:
            return -1.0, 1.0
        v_min = min(values)
        v_max = max(values)
        span = v_max - v_min
        if span < 1e-6:
            baseline = max(1.0, 0.5 * max(abs(v_min), abs(v_max), 1.0))
            return v_min - baseline, v_max + baseline
        padding = max(0.15 * span, 0.05 * max(abs(v_min), abs(v_max), 1.0))
        return v_min - padding, v_max + padding

    def data_to_screen(self, progress: float, value: float) -> QPointF:
        x0, y0, w, h = self._content_rect()
        v_min, v_max = self._value_bounds()
        span = max(1e-6, v_max - v_min)
        x = x0 + np.clip(progress, 0.0, 1.0) * w
        y = y0 + (1.0 - (value - v_min) / span) * h
        return QPointF(x, y)

    def screen_to_data(self, point: QPointF) -> Tuple[float, float]:
        x0, y0, w, h = self._content_rect()
        v_min, v_max = self._value_bounds()
        span = max(1e-6, v_max - v_min)
        progress = (point.x() - x0) / w
        progress = min(1.0, max(0.0, progress))
        value = v_max - (point.y() - y0) / h * span
        return progress, value

    # Painting ------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QBrush(QColor(30, 30, 30)))

        x0, y0, w, h = self._content_rect()
        v_min, v_max = self._value_bounds()

        # Axis box
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(x0, y0, w, h)

        # Grid lines
        painter.setPen(QPen(QColor(80, 80, 80)))
        for i in range(1, 5):
            x = x0 + i * w / 5.0
            y = y0 + i * h / 5.0
            painter.drawLine(QPointF(x, y0), QPointF(x, y0 + h))
            painter.drawLine(QPointF(x0, y), QPointF(x0 + w, y))

        # Labels
        painter.setPen(QPen(QColor(220, 220, 220)))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(x0 - 30, y0 + h + 20, "Progress")
        units_label = f"Value ({self.units})" if self.units else "Value"
        painter.drawText(x0 - 35, y0 - 10, units_label)

        # Plot line
        painter.setPen(QPen(QColor(100, 200, 255), 2))
        if len(self.points) >= 2:
            sorted_points = sorted(self.points, key=lambda item: item[0])
            last = None
            for p, v in sorted_points:
                screen_pt = self.data_to_screen(p, v)
                if last is not None:
                    painter.drawLine(last, screen_pt)
                last = screen_pt

        # Draw points
        painter.setBrush(QColor(255, 140, 0))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for p, v in self.points:
            screen_pt = self.data_to_screen(p, v)
            radius = 6 if p in (0.0, 1.0) else 5
            painter.drawEllipse(screen_pt, radius, radius)

        # Value scale annotations
        painter.setPen(QPen(QColor(160, 160, 160)))
        for frac in np.linspace(0, 1, 5):
            value = float(v_min + frac * (v_max - v_min))
            y = float(y0 + (1 - frac) * h)
            painter.drawText(QPointF(float(x0 - 35), y + 5.0), f"{value:.2f}")

    # Interaction ---------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = event.pos()
        if self.mode == "add":
            progress, value = self.screen_to_data(pos)
            self.points.append((progress, value))
            self.points.sort(key=lambda item: item[0])
            self.update()
        elif self.mode == "remove":
            idx = self._find_nearest_point(pos)
            if idx is not None and len(self.points) > 2:
                progress, _ = self.points[idx]
                if progress not in (0.0, 1.0):
                    self.points.pop(idx)
                    self.update()
        elif self.mode == "move":
            idx = self._find_nearest_point(pos)
            if idx is not None:
                self.drag_index = idx

    def mouseMoveEvent(self, event):
        if self.drag_index is None or self.mode != "move":
            return
        progress, value = self.screen_to_data(event.pos())
        p, _ = self.points[self.drag_index]
        if p == 0.0:
            progress = 0.0
        elif p == 1.0:
            progress = 1.0
        self.points[self.drag_index] = (progress, value)
        self.points.sort(key=lambda item: item[0])
        self.drag_index = min(
            range(len(self.points)),
            key=lambda idx: abs(self.points[idx][0] - progress),
        )
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_index = None

    def _find_nearest_point(self, pos):
        min_distance = None
        min_index = None
        for idx, (p, v) in enumerate(self.points):
            screen_pt = self.data_to_screen(p, v)
            dist = (screen_pt - pos).manhattanLength()
            if min_distance is None or dist < min_distance:
                min_distance = dist
                min_index = idx
        if min_distance is not None and min_distance < 25:
            return min_index
        return None


class ReferenceCurveCanvas(QWidget):
    """Read-only plot to show curvature or other reference against progress."""

    def __init__(self, parent=None, label: str = "Reference"):
        super().__init__(parent)
        self.title = label
        self.progress = np.array([])
        self.values = np.array([])
        self.setMinimumHeight(200)
        self.setMinimumWidth(560)

    def set_data(self, progress, values):
        if progress is None or values is None:
            self.progress = np.array([])
            self.values = np.array([])
        else:
            self.progress = np.asarray(progress, dtype=float)
            self.values = np.asarray(values, dtype=float)
        self.update()

    def _content_rect(self):
        margin = 40
        return (
            margin,
            margin,
            max(10, self.width() - 2 * margin),
            max(10, self.height() - 2 * margin),
        )

    def _value_bounds(self):
        if self.values.size == 0:
            return 0.0, 1.0
        v_min = float(np.min(self.values))
        v_max = float(np.max(self.values))
        span = v_max - v_min
        if span < 1e-6:
            baseline = max(1.0, 0.5 * max(abs(v_min), abs(v_max), 1.0))
            return v_min - baseline, v_max + baseline
        padding = max(0.15 * span, 0.05 * max(abs(v_min), abs(v_max), 1.0))
        return v_min - padding, v_max + padding

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QBrush(QColor(25, 25, 25)))

        x0, y0, w, h = self._content_rect()
        v_min, v_max = self._value_bounds()
        span = max(1e-6, v_max - v_min)

        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(x0, y0, w, h)

        painter.setPen(QPen(QColor(80, 80, 80)))
        for i in range(1, 5):
            x = float(x0 + i * w / 5.0)
            y = float(y0 + i * h / 5.0)
            painter.drawLine(QPointF(x, float(y0)), QPointF(x, float(y0 + h)))
            painter.drawLine(QPointF(float(x0), y), QPointF(float(x0 + w), y))

        painter.setPen(QPen(QColor(220, 220, 220)))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(QPointF(float(x0 - 30), float(y0 + h + 20)), "Progress")
        painter.drawText(QPointF(float(x0 - 35), float(y0 - 10)), self.title)

        if self.progress.size >= 2:
            painter.setPen(QPen(QColor(255, 140, 0), 2))
            order = np.argsort(self.progress)
            prog_sorted = self.progress[order]
            val_sorted = self.values[order]
            if prog_sorted[0] > 0.0:
                prog_sorted = np.insert(prog_sorted, 0, 0.0)
                val_sorted = np.insert(val_sorted, 0, val_sorted[0])
            if prog_sorted[-1] < 1.0:
                prog_sorted = np.append(prog_sorted, 1.0)
                val_sorted = np.append(val_sorted, val_sorted[0])
            last_pt = None
            for prog, val in zip(prog_sorted, val_sorted):
                x = x0 + np.clip(prog, 0.0, 1.0) * w
                y = y0 + (1.0 - (val - v_min) / span) * h
                pt = QPointF(x, y)
                if last_pt is not None:
                    painter.drawLine(last_pt, pt)
                last_pt = pt

        painter.setPen(QPen(QColor(160, 160, 160)))
        for frac in np.linspace(0.0, 1.0, 5):
            value = float(v_min + frac * span)
            y_coord = float(y0 + (1.0 - frac) * h)
            painter.drawText(QPointF(float(x0 - 35), y_coord + 5.0), f"{value:.2f}")

class FunctionEditorDialog(QDialog):
    """Dialog presenting controls to edit a ParameterFunction."""

    def __init__(
        self,
        parent,
        parameter_function,
        title: str,
        units: str = "",
        reference=None,
        auto_spacing_params=None,
        auto_spacing_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        flags = self.windowFlags()
        flags |= Qt.Window
        flags |= Qt.WindowMaximizeButtonHint
        flags |= Qt.WindowMinimizeButtonHint
        self.setWindowFlags(flags)
        self.setSizeGripEnabled(True)
        self.resize(780, 640)
        self.parameter_function = parameter_function
        self.canvas = FunctionCanvas(self, units=units)
        self.canvas.set_points(parameter_function.get_control_points())
        self.auto_spacing_callback = auto_spacing_callback
        self._auto_spacing_params = dict(auto_spacing_params or {})

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Left click to add/move points. Choose a mode below."))
        layout.addWidget(self.canvas)

        if reference is not None and reference.get("progress") is not None:
            ref_label = reference.get("label", "Reference")
            layout.addWidget(QLabel(f"{ref_label} vs progress"))
            self.reference_canvas = ReferenceCurveCanvas(self, ref_label)
            self.reference_canvas.set_data(reference.get("progress"), reference.get("values"))
            layout.addWidget(self.reference_canvas)
        else:
            self.reference_canvas = None

        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.add_button = QRadioButton("Add")
        self.move_button = QRadioButton("Move")
        self.remove_button = QRadioButton("Remove")
        self.mode_group.addButton(self.add_button)
        self.mode_group.addButton(self.move_button)
        self.mode_group.addButton(self.remove_button)
        self.add_button.setChecked(True)
        self.canvas.set_mode("add")
        self.mode_group.buttonClicked.connect(self._mode_changed)
        mode_layout.addWidget(self.add_button)
        mode_layout.addWidget(self.move_button)
        mode_layout.addWidget(self.remove_button)
        layout.addLayout(mode_layout)

        self.reset_button = QPushButton("Reset to Constant")
        self.reset_button.clicked.connect(self._reset_points)

        self.info_label = QLabel("Endpoints stay at progress 0 and 1.")

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.reset_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.info_label)
        layout.addLayout(bottom_layout)

        if self._auto_spacing_params:
            layout.addWidget(QLabel("Curvature-based auto spacing parameters"))
            form = QFormLayout()
            multiplier = float(self._auto_spacing_params.get("multiplier", 0.7))
            minimum = float(self._auto_spacing_params.get("min", 1.0))
            maximum = float(self._auto_spacing_params.get("max", 5.0))

            self.auto_spacing_multiplier_edit = QLineEdit(f"{multiplier:.3f}")
            self.auto_spacing_min_edit = QLineEdit(f"{minimum:.3f}")
            self.auto_spacing_max_edit = QLineEdit(f"{maximum:.3f}")

            self.auto_spacing_multiplier_edit.editingFinished.connect(self._sync_auto_spacing_params)
            self.auto_spacing_min_edit.editingFinished.connect(self._sync_auto_spacing_params)
            self.auto_spacing_max_edit.editingFinished.connect(self._sync_auto_spacing_params)

            form.addRow("Multiplier", self.auto_spacing_multiplier_edit)
            form.addRow("Minimum spacing (m)", self.auto_spacing_min_edit)
            form.addRow("Maximum spacing (m)", self.auto_spacing_max_edit)
            layout.addLayout(form)

            self.auto_spacing_formula_label = QLabel()
            self.auto_spacing_formula_label.setWordWrap(True)
            layout.addWidget(self.auto_spacing_formula_label)
            self._update_auto_spacing_formula_label()

        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

    def _mode_changed(self, button):
        mode_map = {
            self.add_button: "add",
            self.move_button: "move",
            self.remove_button: "remove",
        }
        self.canvas.set_mode(mode_map.get(button, "add"))

    def _reset_points(self):
        default_value = self.parameter_function.default_value
        self.canvas.set_points([(0.0, default_value), (1.0, default_value)])

    def accept(self):
        points = self.canvas.get_points()
        self.parameter_function.set_control_points(points)
        if self._auto_spacing_params and self.auto_spacing_callback is not None:
            params = self._collect_auto_spacing_params()
            self.auto_spacing_callback(
                params["multiplier"], params["min"], params["max"]
            )
        super().accept()

    # Auto spacing helpers -------------------------------------------------
    def _parse_auto_value(self, edit, fallback, positive=False):
        text = edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = fallback
        if positive and value <= 0:
            value = fallback
        edit.setText(f"{value:.3f}")
        return value

    def _collect_auto_spacing_params(self):
        if not self._auto_spacing_params:
            return {}
        multiplier = self._parse_auto_value(
            self.auto_spacing_multiplier_edit,
            self._auto_spacing_params.get("multiplier", 0.7),
            positive=True,
        )
        minimum = self._parse_auto_value(
            self.auto_spacing_min_edit,
            self._auto_spacing_params.get("min", 1.0),
            positive=True,
        )
        maximum = self._parse_auto_value(
            self.auto_spacing_max_edit,
            self._auto_spacing_params.get("max", 5.0),
            positive=True,
        )
        if maximum < minimum:
            maximum = minimum
            self.auto_spacing_max_edit.setText(f"{maximum:.3f}")
        params = {"multiplier": multiplier, "min": minimum, "max": maximum}
        self._auto_spacing_params.update(params)
        self._update_auto_spacing_formula_label()
        return params

    def _sync_auto_spacing_params(self):
        self._collect_auto_spacing_params()

    def _update_auto_spacing_formula_label(self):
        if not self._auto_spacing_params:
            return
        multiplier = float(self._auto_spacing_params.get("multiplier", 0.7))
        minimum = float(self._auto_spacing_params.get("min", 1.0))
        maximum = float(self._auto_spacing_params.get("max", 5.0))
        self.auto_spacing_formula_label.setText(
            (
                "Spacing = clamp({:.3f} / κ, {:.3f}, {:.3f})\n"
                "κ is curvature (1/m); spacing is in meters."
            ).format(multiplier, minimum, maximum)
        )
