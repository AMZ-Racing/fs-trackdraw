from fs_trackdraw_qt import FSTrackDraw
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication([])
    window = FSTrackDraw()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()