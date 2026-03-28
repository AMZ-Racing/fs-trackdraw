import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from trackdraw.window import TrackDrawWindow


def main():
    app = QApplication([])
    icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TrackDraw_Logo.png"))
    icon = QIcon(icon_path)
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = TrackDrawWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
