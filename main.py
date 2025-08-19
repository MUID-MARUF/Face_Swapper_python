from PyQt5.QtWidgets import QApplication
from gui import FaceSwapGUI
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceSwapGUI()
    window.show()
    sys.exit(app.exec_())
