from PyQt5 import QtWidgets
import sys
import controller

    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ctl = controller.Controller()
    sys.exit(app.exec_())