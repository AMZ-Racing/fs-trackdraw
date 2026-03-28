from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget


class TrackCanvas(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.pan_start = QPointF(0, 0)
        self.is_panning = False

        self.background_kind = "grid"
        self.background_image = QImage()
        self.background_width = 1600
        self.background_height = 1200
        self.px_per_m = 10.0
        self.grid_spacing_m = 1.0

        self.allowed_area = []
        self.control_points = []
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.left_cones = None
        self.right_cones = None
        self.edit_target = "area"
        self.transform_gizmo = None

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def set_background(
        self,
        kind: str,
        px_per_m: float,
        image_path: str = "",
        width_px: int = 1600,
        height_px: int = 1200,
        grid_spacing_m: float = 1.0,
    ):
        self.background_kind = kind
        self.px_per_m = float(px_per_m)
        self.grid_spacing_m = max(0.1, float(grid_spacing_m))
        self.background_image = QImage()
        if kind == "location" and image_path:
            image = QImage(image_path)
            if not image.isNull():
                self.background_image = image
                self.background_width = image.width()
                self.background_height = image.height()
            else:
                self.background_width = width_px
                self.background_height = height_px
        else:
            self.background_width = width_px
            self.background_height = height_px
        self.update()

    def background_dimensions(self) -> Tuple[int, int]:
        return int(self.background_width), int(self.background_height)

    def update_scene(
        self,
        allowed_area,
        control_points,
        centerline,
        left_boundary,
        right_boundary,
        left_cones,
        right_cones,
        edit_target: str,
        transform_gizmo=None,
    ):
        self.allowed_area = list(allowed_area)
        self.control_points = list(control_points)
        self.centerline = centerline
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.left_cones = left_cones
        self.right_cones = right_cones
        self.edit_target = edit_target
        self.transform_gizmo = transform_gizmo
        self.update()

    def _background_rect(self) -> QRectF:
        bg_w, bg_h = self.background_dimensions()
        if bg_w <= 0 or bg_h <= 0:
            return QRectF(0, 0, float(self.width()), float(self.height()))
        scale = min(self.width() / bg_w, self.height() / bg_h)
        draw_w = bg_w * scale
        draw_h = bg_h * scale
        offset_x = (self.width() - draw_w) / 2.0
        offset_y = (self.height() - draw_h) / 2.0
        return QRectF(offset_x, offset_y, draw_w, draw_h)

    def _map_points_to_scene(self, points: Iterable[Sequence[float]], rect: QRectF | None = None):
        rect = rect or self._background_rect()
        bg_w, bg_h = self.background_dimensions()
        if bg_w <= 0 or bg_h <= 0:
            return []

        arr = np.asarray(points, dtype=float)
        if arr.size == 0:
            return []
        arr = arr.reshape(-1, 2)

        scale_x = rect.width() / bg_w
        scale_y = rect.height() / bg_h
        mapped = np.empty_like(arr)
        mapped[:, 0] = rect.x() + arr[:, 0] * scale_x
        mapped[:, 1] = rect.y() + arr[:, 1] * scale_y
        return [QPointF(float(x), float(y)) for x, y in mapped]

    def map_to_scene(self, point: Sequence[float]) -> QPointF:
        mapped = self._map_points_to_scene([point])
        if not mapped:
            return QPointF(0.0, 0.0)
        return mapped[0]

    def scene_to_map(self, point: QPointF) -> QPointF:
        rect = self._background_rect()
        bg_w, bg_h = self.background_dimensions()
        if bg_w <= 0 or bg_h <= 0:
            return QPointF(0.0, 0.0)
        map_x = (point.x() - rect.x()) * bg_w / max(rect.width(), 1.0)
        map_y = (point.y() - rect.y()) * bg_h / max(rect.height(), 1.0)
        return QPointF(map_x, map_y)

    def _screen_to_scene(self, screen_point: QPointF) -> QPointF:
        return QPointF(
            (screen_point.x() - self.pan.x()) / self.zoom,
            (screen_point.y() - self.pan.y()) / self.zoom,
        )

    def screen_to_map(self, screen_point: QPointF) -> QPointF:
        return self.scene_to_map(self._screen_to_scene(screen_point))

    def _scene_to_screen(self, scene_point: QPointF) -> QPointF:
        return QPointF(
            scene_point.x() * self.zoom + self.pan.x(),
            scene_point.y() * self.zoom + self.pan.y(),
        )

    def _draw_grid(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, QColor(249, 246, 236))
        bg_w, bg_h = self.background_dimensions()
        minor_step_px = max(1.0, self.grid_spacing_m * self.px_per_m)
        major_step_px = minor_step_px * 5.0
        scale_x = rect.width() / max(bg_w, 1)
        scale_y = rect.height() / max(bg_h, 1)

        minor_pen = QPen(QColor(223, 220, 210), 1)
        major_pen = QPen(QColor(186, 182, 170), 1)

        x = 0.0
        while x <= bg_w:
            scene_x = rect.x() + x * scale_x
            painter.setPen(major_pen if abs((x / major_step_px) - round(x / major_step_px)) < 1e-6 else minor_pen)
            painter.drawLine(QPointF(scene_x, rect.top()), QPointF(scene_x, rect.bottom()))
            x += minor_step_px

        y = 0.0
        while y <= bg_h:
            scene_y = rect.y() + y * scale_y
            painter.setPen(major_pen if abs((y / major_step_px) - round(y / major_step_px)) < 1e-6 else minor_pen)
            painter.drawLine(QPointF(rect.left(), scene_y), QPointF(rect.right(), scene_y))
            y += minor_step_px

        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawRect(rect)

    def _draw_polyline(
        self,
        painter: QPainter,
        points: Iterable[Sequence[float]],
        color: QColor,
        rect: QRectF,
        width: int = 2,
        closed: bool = False,
        style=Qt.SolidLine,
    ):
        pts = self._map_points_to_scene(points, rect)
        if len(pts) < 2:
            return
        poly = QPolygonF(pts)
        if closed:
            poly.append(pts[0])
        pen = QPen(color, width)
        pen.setStyle(style)
        painter.setPen(pen)
        painter.drawPolyline(poly)

    def _draw_points(self, painter: QPainter, points: Iterable[Sequence[float]], fill: QColor, rect: QRectF, radius: int = 5):
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(fill)
        for scene in self._map_points_to_scene(points, rect):
            painter.drawEllipse(scene, radius, radius)

    def _draw_arrow_handle(self, painter: QPainter, start: QPointF, end: QPointF, color: QColor):
        painter.setPen(QPen(color, 2))
        painter.drawLine(start, end)
        direction = np.array([end.x() - start.x(), end.y() - start.y()], dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return
        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=float)
        head_size = 8.0
        base = np.array([end.x(), end.y()], dtype=float) - direction * head_size
        left = base + normal * (0.55 * head_size)
        right = base - normal * (0.55 * head_size)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([
            QPointF(float(end.x()), float(end.y())),
            QPointF(float(left[0]), float(left[1])),
            QPointF(float(right[0]), float(right[1])),
        ]))

    def _draw_transform_gizmo(self, painter: QPainter):
        gizmo = self.transform_gizmo
        if not gizmo:
            return

        center = self.map_to_scene(gizmo["center"])
        x_axis = self.map_to_scene(gizmo["translate_x"])
        y_axis = self.map_to_scene(gizmo["translate_y"])
        rotate = self.map_to_scene(gizmo["rotate"])

        self._draw_arrow_handle(painter, center, x_axis, QColor(191, 66, 66))
        self._draw_arrow_handle(painter, center, y_axis, QColor(52, 109, 191))

        painter.setPen(QPen(QColor(76, 76, 76), 2, Qt.DashLine))
        painter.drawLine(center, rotate)
        radius = float(np.hypot(rotate.x() - center.x(), rotate.y() - center.y()))
        ring = QRectF(center.x() - radius, center.y() - radius, 2.0 * radius, 2.0 * radius)
        painter.drawArc(ring, 35 * 16, 290 * 16)

        painter.setPen(QPen(QColor(35, 35, 35), 1))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRect(QRectF(center.x() - 7.0, center.y() - 7.0, 14.0, 14.0))
        painter.setBrush(QColor(191, 66, 66))
        painter.drawEllipse(x_axis, 5, 5)
        painter.setBrush(QColor(52, 109, 191))
        painter.drawEllipse(y_axis, 5, 5)
        painter.setBrush(QColor(91, 44, 143))
        painter.drawEllipse(rotate, 6, 6)

    def transform_handle_at(self, screen_point: QPointF):
        gizmo = self.transform_gizmo
        if not gizmo:
            return None

        scene_point = self._screen_to_scene(screen_point)
        handles = {
            "track_translate_free": self.map_to_scene(gizmo["center"]),
            "track_translate_x": self.map_to_scene(gizmo["translate_x"]),
            "track_translate_y": self.map_to_scene(gizmo["translate_y"]),
            "track_rotate": self.map_to_scene(gizmo["rotate"]),
        }
        thresholds = {
            "track_translate_free": 12.0,
            "track_translate_x": 12.0,
            "track_translate_y": 12.0,
            "track_rotate": 14.0,
        }
        for kind, handle in handles.items():
            if np.hypot(scene_point.x() - handle.x(), scene_point.y() - handle.y()) <= thresholds[kind]:
                return kind
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(235, 235, 235))
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)

        rect = self._background_rect()
        if self.background_kind == "location" and not self.background_image.isNull():
            painter.drawImage(rect, self.background_image)
        else:
            self._draw_grid(painter, rect)

        if len(self.allowed_area) >= 3:
            polygon = QPolygonF(self._map_points_to_scene(self.allowed_area, rect))
            painter.setPen(QPen(QColor(180, 65, 0), 2))
            painter.setBrush(QColor(255, 177, 110, 70))
            painter.drawPolygon(polygon)
        elif len(self.allowed_area) >= 2:
            self._draw_polyline(painter, self.allowed_area, QColor(180, 65, 0), rect, width=2)
        self._draw_points(
            painter,
            self.allowed_area,
            QColor(255, 140, 0) if self.edit_target == "area" else QColor(209, 157, 88),
            rect,
            radius=5,
        )

        if self.centerline is not None:
            self._draw_polyline(painter, self.centerline, QColor(16, 132, 86), rect, width=2, closed=False, style=Qt.DashLine)
        if self.left_boundary is not None:
            self._draw_polyline(painter, self.left_boundary, QColor(0, 76, 255), rect, width=2, closed=False)
        if self.right_boundary is not None:
            self._draw_polyline(painter, self.right_boundary, QColor(222, 186, 0), rect, width=2, closed=False)

        if self.left_cones is not None:
            if self.edit_target == "cones":
                self._draw_polyline(painter, self.left_cones, QColor(0, 76, 255), rect, width=1, closed=True)
            self._draw_points(painter, self.left_cones, QColor(0, 76, 255), rect, radius=3)
        if self.right_cones is not None:
            if self.edit_target == "cones":
                self._draw_polyline(painter, self.right_cones, QColor(222, 186, 0), rect, width=1, closed=True)
            self._draw_points(painter, self.right_cones, QColor(222, 186, 0), rect, radius=3)

        if self.control_points:
            self._draw_points(
                painter,
                self.control_points,
                QColor(48, 176, 96) if self.edit_target == "track" else QColor(108, 156, 118),
                rect,
                radius=5,
            )

        self._draw_transform_gizmo(painter)

    def wheelEvent(self, event):
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1.0 / 1.1
        mouse_pos = event.pos()
        old_scene = self.screen_to_map(mouse_pos)
        self.zoom = max(0.15, min(self.zoom * zoom_factor, 20.0))
        new_scene_pos = self.map_to_scene((old_scene.x(), old_scene.y()))
        new_screen_pos = QPointF(new_scene_pos.x() * self.zoom + self.pan.x(), new_scene_pos.y() * self.zoom + self.pan.y())
        delta = mouse_pos - new_screen_pos
        self.pan += delta
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            handle = self.transform_handle_at(event.pos())
            self.parent.handle_canvas_click(self.screen_to_map(event.pos()), handle)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.pan_start
            self.pan += delta
            self.pan_start = event.pos()
            self.update()
            return
        if self.parent.has_active_drag():
            self.parent.handle_canvas_drag(self.screen_to_map(event.pos()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        self.parent.handle_canvas_release(self.screen_to_map(event.pos()))
