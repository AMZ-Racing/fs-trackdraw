import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF


class TrackCanvas(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Zoom/pan variables
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.pan_start = QPointF(0, 0)
        self.is_panning = False
        
        # Initialize scaling factors
        self.map_scale_x = 1.0
        self.map_scale_y = 1.0
        self.map_offset_x = 0
        self.map_offset_y = 0
        
        # Load the satellite image
        self.sat_image = QImage(parent.fpath_location_sat_img)
        self.sat_pixmap = QPixmap.fromImage(self.sat_image)
        
        # Drawing elements
        self.control_points = []
        self.centerline = None
        self.left_boundary = None
        self.right_boundary = None
        self.boundaries_swapped = False
        self.left_cones = None
        self.right_cones = None
        self.centerline_alert_segments = []
        self.left_alert_segments = []
        self.right_alert_segments = []
        self.curvature_green_segments = []
        self.curvature_red_segments = []

        self.barrier_polygon = None  # Barrier polygon
        self.barrier_offset_polygon = None  # Offset polygon inside the barrier

        # colors for drawing
        self.left_color = QColor(0, 0, 255)
        self.right_color = QColor(255, 255, 0)
        self.barrier_color = QColor(255, 0, 0)
        self.barrier_offset_color = QColor(0, 255, 0)

    def wheelEvent(self, event):
        # Zoom factor (10% per wheel step)
        zoom_factor = 1.1
        if event.angleDelta().y() < 0:
            zoom_factor = 1 / zoom_factor

        # Get mouse position before zoom
        mouse_pos = event.pos()
        old_scene_pos = self.screen_to_map(mouse_pos)
        
        # Apply zoom
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(self.zoom, 10.0))  # Limit zoom range
        
        # Get mouse position after zoom
        new_screen_pos = self.map_to_screen(old_scene_pos)
        
        # Adjust pan to zoom toward mouse
        delta = mouse_pos - new_screen_pos
        self.pan += delta
        
        self.update()
        
    def transform_point(self, map_x, map_y):
        """Convert from map coordinates to display coordinates"""
        return (map_x * self.map_scale_x + self.map_offset_x,
                map_y * self.map_scale_y + self.map_offset_y)

    def inverse_transform_point(self, display_x, display_y):
        """Convert from display coordinates to map coordinates"""
        return ((display_x - self.map_offset_x) / self.map_scale_x,
                (display_y - self.map_offset_y) / self.map_scale_y)
    
    def transform_polygon(self, polygon):
        """Transform a polygon from map to display coordinates"""
        return [self.transform_point(p[0], p[1]) for p in polygon]
    
    def screen_to_map(self, screen_point):
        """Convert screen coordinates to map coordinates"""
        # Remove pan and zoom, then remove image scaling
        image_x = (screen_point.x() - self.pan.x()) / self.zoom
        image_y = (screen_point.y() - self.pan.y()) / self.zoom
        map_x = (image_x - self.map_offset_x) / self.map_scale_x
        map_y = (image_y - self.map_offset_y) / self.map_scale_y
        return QPointF(map_x, map_y)

    def map_to_screen(self, map_point):
        """Convert map coordinates to screen coordinates"""
        # Apply image scaling, then zoom and pan
        image_x = map_point.x() * self.map_scale_x + self.map_offset_x
        image_y = map_point.y() * self.map_scale_y + self.map_offset_y
        screen_x = image_x * self.zoom + self.pan.x()
        screen_y = image_y * self.zoom + self.pan.y()
        return QPointF(int(screen_x), int(screen_y))
    
    def screen_to_scene(self, screen_point):
        """Convert screen coordinates to image coordinates (including zoom/pan)"""
        return QPointF(
            (screen_point.x() - self.pan.x()) / self.zoom,
            (screen_point.y() - self.pan.y()) / self.zoom
        )

    def scene_to_screen(self, scene_point):
        """Convert image coordinates to screen coordinates (including zoom/pan)"""
        return QPointF(
            scene_point.x() * self.zoom + self.pan.x(),
            scene_point.y() * self.zoom + self.pan.y()
        )

    def update_drawing(self, control_points, centerline, left_boundary, right_boundary, 
                      boundaries_swapped, barrier_polygon, barrier_offset_polygon,
                      left_cones=None, right_cones=None,
                      centerline_warn_segments=None, left_warn_segments=None, right_warn_segments=None,
                      curvature_green_segments=None, curvature_red_segments=None):
        """Update the drawing with new data"""
        self.control_points = control_points
        self.centerline = centerline
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.boundaries_swapped = boundaries_swapped
        self.barrier_polygon = barrier_polygon
        self.barrier_offset_polygon = barrier_offset_polygon
        self.left_cones = left_cones
        self.right_cones = right_cones
        self.centerline_alert_segments = centerline_warn_segments or []
        self.left_alert_segments = left_warn_segments or []
        self.right_alert_segments = right_warn_segments or []
        self.curvature_green_segments = curvature_green_segments or []
        self.curvature_red_segments = curvature_red_segments or []
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Apply zoom/pan transformation
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)

        # Draw scaled satellite image
        scaled_pixmap = self.sat_pixmap.scaled(
            self.width(), self.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(0, 0, scaled_pixmap)

        # Recalculate map scaling factors based on current window size and image size
        self.map_scale_x = scaled_pixmap.width() / self.sat_image.width()
        self.map_scale_y = scaled_pixmap.height() / self.sat_image.height()
        
        # Draw control points (in display coordinates)
        for i, pt in enumerate(self.control_points):
            display_pt = QPointF(*self.transform_point(pt.x(), pt.y()))
            if i == 0 and len(self.control_points) > 2:
                painter.setBrush(QColor(255, 0, 0))  # Red for first point
            elif i == len(self.control_points) - 1 and len(self.control_points) > 2:
                painter.setBrush(QColor(0, 0, 255))  # Blue for last point
            else:
                painter.setBrush(QColor(0, 255, 0))  # Green for others
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(display_pt, 5, 5)

        # Draw centerline (transform from map to display coords)
        if self.centerline and len(self.centerline) > 1:
            pen = QPen(QColor(0, 255, 0), 2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            
            # Convert all centerline points to display coordinates
            display_points = []
            for pt in self.centerline:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:  # Assume raw coordinates
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            
            # Draw connected lines
            for i in range(1, len(display_points)):
                painter.drawLine(display_points[i-1], display_points[i])

            if self.centerline_alert_segments:
                alert_pen = QPen(QColor(255, 0, 0), 3)
                alert_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(alert_pen)
                for p0, p1 in self.centerline_alert_segments:
                    pt0 = QPointF(*self.transform_point(p0[0], p0[1]))
                    pt1 = QPointF(*self.transform_point(p1[0], p1[1]))
                    painter.drawLine(pt0, pt1)

        # Draw boundaries (transform from map to display coords)
        if self.left_boundary and len(self.left_boundary) > 1:
            painter.setPen(QPen(self.left_color, 2))
            
            # Convert boundary points to display coordinates
            display_points = []
            for pt in self.left_boundary:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:  # Assume raw coordinates
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            
            painter.drawPolyline(QPolygonF(display_points))

            if self.left_alert_segments:
                alert_pen = QPen(QColor(255, 0, 0), 3)
                alert_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(alert_pen)
                for p0, p1 in self.left_alert_segments:
                    painter.drawLine(
                        QPointF(*self.transform_point(p0[0], p0[1])),
                        QPointF(*self.transform_point(p1[0], p1[1]))
                    )

        if self.right_boundary and len(self.right_boundary) > 1:
            painter.setPen(QPen(self.right_color, 2))
            
            # Convert boundary points to display coordinates
            display_points = []
            for pt in self.right_boundary:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:  # Assume raw coordinates
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            
            painter.drawPolyline(QPolygonF(display_points))

            if self.right_alert_segments:
                alert_pen = QPen(QColor(255, 0, 0), 3)
                alert_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(alert_pen)
                for p0, p1 in self.right_alert_segments:
                    painter.drawLine(
                        QPointF(*self.transform_point(p0[0], p0[1])),
                        QPointF(*self.transform_point(p1[0], p1[1]))
                    )

        # Draw curvature analysis path segments
        if self.curvature_green_segments:
            pen_green = QPen(QColor(0, 100, 0), 3)
            pen_green.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_green)
            for p0, p1 in self.curvature_green_segments:
                painter.drawLine(
                    QPointF(*self.transform_point(p0.x(), p0.y())),
                    QPointF(*self.transform_point(p1.x(), p1.y()))
                )

        if self.curvature_red_segments:
            pen_red = QPen(QColor(220, 0, 0), 3)
            pen_red.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_red)
            for p0, p1 in self.curvature_red_segments:
                painter.drawLine(
                    QPointF(*self.transform_point(p0.x(), p0.y())),
                    QPointF(*self.transform_point(p1.x(), p1.y()))
                )


        # Draw cones with precomputed positions
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        if self.left_cones is not None:
            painter.setBrush(self.left_color)
            for pt in self.left_cones:
                if isinstance(pt, QPointF):
                    display_pt = QPointF(*self.transform_point(pt.x(), pt.y()))
                else:
                    display_pt = QPointF(*self.transform_point(pt[0], pt[1]))
                painter.drawEllipse(display_pt, 3, 3)

        if self.right_cones is not None:
            painter.setBrush(self.right_color)
            for pt in self.right_cones:
                if isinstance(pt, QPointF):
                    display_pt = QPointF(*self.transform_point(pt.x(), pt.y()))
                else:
                    display_pt = QPointF(*self.transform_point(pt[0], pt[1]))
                painter.drawEllipse(display_pt, 3, 3)

        # Draw barriers (transform from map to display coords)
        if self.barrier_polygon is not None:
            # Draw barrier polygon
            for i, pt in enumerate(self.barrier_polygon):
                display_pt = QPointF(*self.transform_point(pt.x(), pt.y()))
                if i == 0 and len(self.barrier_polygon) > 2:
                    painter.setBrush(QColor(0, 255, 0))  # Red for first point
                elif i == len(self.barrier_polygon) - 1 and len(self.barrier_polygon) > 2:
                    painter.setBrush(QColor(0, 0, 255))  # Blue for last point
                else:
                    painter.setBrush(self.barrier_color)  # Red for others
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawEllipse(display_pt, 5, 5)
            # Draw barrier polygon outline
            painter.setPen(QPen(self.barrier_color, 2))
            display_points = []
            for pt in self.barrier_polygon:
                if isinstance(pt, QPointF):
                    display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                else:
                    display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
            painter.drawPolyline(QPolygonF(display_points))
            # Draw offset polygon
            if self.barrier_offset_polygon is not None:
                painter.setPen(QPen(self.barrier_offset_color, 2))
                display_points = []
                for pt in self.barrier_offset_polygon:
                    if isinstance(pt, QPointF):
                        display_points.append(QPointF(*self.transform_point(pt.x(), pt.y())))
                    else:
                        display_points.append(QPointF(*self.transform_point(pt[0], pt[1])))
                painter.drawPolyline(QPolygonF(display_points))
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.RightButton:
            # Convert click to scene coordinates
            scene_pos = self.screen_to_scene(event.pos())
            # Then convert to map coordinates
            map_x, map_y = self.inverse_transform_point(scene_pos.x(), scene_pos.y())
            self.parent.handle_canvas_rightclick(QPointF(map_x, map_y))
        else:
            # Convert click to scene coordinates
            scene_pos = self.screen_to_scene(event.pos())
            # Then convert to map coordinates
            map_x, map_y = self.inverse_transform_point(scene_pos.x(), scene_pos.y())
            self.parent.handle_canvas_click(QPointF(map_x, map_y))

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.pan_start
            self.pan += delta
            self.pan_start = event.pos()
            self.update()
        elif self.parent.dragging:
            map_pos = self.screen_to_map(event.pos())
            self.parent.handle_canvas_drag(QPointF(map_pos.x(), map_pos.y()))
        elif self.parent.dragging_barrier:
            map_pos = self.screen_to_map(event.pos())
            self.parent.handle_canvas_drag(QPointF(map_pos.x(), map_pos.y()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            self.parent.handle_canvas_release(event.pos())
