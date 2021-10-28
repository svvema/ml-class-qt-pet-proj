import time, sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QStyledItemDelegate, QLineEdit
from PyQt5 import uic
import design
import os
import numpy as np
import get_osci_func, ML_inst
# import pyqtspinner
# from pyqtspinner.spinner import WaitingSpinner
import pandas as pd
from PyQt5.QtGui import QFont

class Ui_MainWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):

    def __init__(self):
        # QtWidgets.QMainWindow.__init__(self)
        # self.ui = uic.loadUi('test.ui',self)
        super().__init__()
        self.setupUi(self)

        self.path = os.getcwd()

        if not os.path.exists(self.path + r'\settings'):
            os.makedirs(self.path + r'\settings')
            self.path = self.path + r'\settings'
        else:
            self.path = self.path + r'\settings'

        self.progressBar.setValue(0)
        self.progressBar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                }

                QProgressBar::chunk {
                    background-color: #37DA7E;
                    width: 20px;
                }""")
        # self.start_learn2 = QtWidgets.QPushButton(self.groupBox_4)
        # self.gridLayout_6.addWidget(self.start_learn2, 0, 1, 1, 1)
        # self.spinner = WaitingSpinner(self.groupBox_4)
        # self.gridLayout_6.addWidget(self.spinner, 0, 2, 1, 1)

        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.path_label.setText(self.path)  # путь текущей папки запуска
        self.load_data()  # загружаем данные по пути рабочей папки
        self.range_chooser_up.setCurrentIndex(9)  # ставим дальность "от" равную первой позиции списка
        delegate = ReadOnlyDelegate(self.tableWidget)
        self.tableWidget.setItemDelegateForColumn(0,delegate)
        self.tableWidget.verticalHeader().setVisible(False)

        if os.path.isfile(self.path + '/options.ini'):
            with open(self.path + '/options.ini', 'r') as f:
                self.ini_lines = f.readlines()
                ip_text = self.ini_lines[0].rstrip()
                self.osci_path = "TCPIP::" + ip_text + "::INSTR"
                self.osci_adress.setText(ip_text)
        else:
            with open(self.path + '/options.ini', 'w') as f:
                # self.osci_path = "TCPIP::" + self.osci_adress.text() + "::INSTR"
                self.osci_path = self.osci_adress.text()
                f.write(self.osci_path)

          # адресс осциллографа по сети
        self.osci_pause = True  # инициализируем состояние осциллографа как на паузе
        self.osci = get_osci_func.osci_func(self.path, self.osci_path, self.osci_pause)  # подключаем интерфейс осциллографа и его функций

        self.connect_osci()  # пробуем подключиться к осциллографу при запуске программы

        #### Подключаем кнопки ####
        #### Tab Снимаем ####
        self.osci_reconnect.clicked.connect(self.reconnect_osci)  # сигнал на переподключение к осциллографу
        self.take_osci.clicked.connect(lambda: self.get_osci())  # сигнал для запуска процедуры сбора осциллограмм
        self.toolButton.clicked.connect(self.browse_folder)  # сигнал для выбора рабочей папки
        self.table_add_row.clicked.connect(self.addRowTable)  # сигнал на добавление новой строки в таблицу
        self.table_remove_row.clicked.connect(self.removeRowTable)  # сигнал на удаление последней строки в таблице
        self.save_table.clicked.connect(self.saveTable)
        self.start_learn.clicked.connect(self.startLearn)
        self.class_chooser.currentIndexChanged.connect(self.combobox_changed)
        #### Tab Смотрим ####


        if os.path.isfile(self.path + '/models/conf.txt'):
            self.init_ml_predict()
        else:
            self.start_predict.setEnabled(False)

        self.start_predict.clicked.connect(self.startPredict)
        self.predict_c.setText('--------------')
        self.predict_r.setText('--------------')
        # self.font_resize(self.predict_c, self.predict_r)
        self.resizeEvent = self.font_resize_event

    def init_ml_predict(self):
        self.osci_pause = True
        self.ml_predict = predictThread(self.path, self.osci)
        self.ml_predict.setTerminationEnabled(True)
        self.ml_predict.class_signal.connect(self.class_signal)
        self.ml_predict.range_signal.connect(self.range_signal)
        self.start_predict.setEnabled(True)

    def item_alignment(self, value):
        item = QTableWidgetItem(value)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        return item

    def browse_folder(self):
        self.path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку")

        self.path_label.setText(self.path)

    def font_resize_event(self, event):
         self.font_resize(self.predict_c, self.predict_r)


    def load_data(self):
        self.class_chooser.clear()
        if os.path.isfile(self.path + '/class_db.npy'):
            self.class_db = np.load(self.path + '/class_db.npy', allow_pickle='TRUE').item()
            numrows = len(self.class_db)
            self.tableWidget.setRowCount(numrows)
            keys_list = list(self.class_db.keys())
            values_list = list(self.class_db.values())
            for row in range(numrows):
                # item = QTableWidgetItem(keys_list[row])
                # item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.tableWidget.setItem(row, 0, self.item_alignment(keys_list[row]))
                # item = QTableWidgetItem(values_list[row])
                # item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.tableWidget.setItem(row, 1, self.item_alignment(values_list[row]))
                self.class_chooser.addItem(str(keys_list[row]))
            self.class_name_label.setText(self.tableWidget.item(0, 1).text())

    def connect_osci(self):
        print('connect_osci', self.osci_path)
        try:
            self.osci.connect()

            self.osci_status_label_now.setText('Подключен')
            self.osci_status_label_now.setStyleSheet("color: green;")
        except:
            # self.osci.disconnect()
            self.osci_status_label_now.setText('Отключен')
            self.osci_status_label_now.setStyleSheet("color: red;")
            print('Осцилла нет')

    def reconnect_osci(self):
        ip_text = self.osci_adress.text()
        self.ini_lines[0] = ip_text + '\n'
        with open(self.path + '/options.ini', 'w') as f:
            for line in self.ini_lines:
                f.write(line)

        self.osci_path = "TCPIP::" + ip_text + "::INSTR"

        self.connect_osci()

    def get_osci(self):
        self.ml_predict.stop()  # останавливаем существующий тред предсказаний
        osci = self.osci
        current_class = int(self.class_chooser.currentText())
        bottom = int(self.range_chooser_bottom.currentText())
        up = int(self.range_chooser_up.currentText())
        steps = int(self.steps.value())

        self.thread = progressThread(osci, current_class,bottom, up, steps)
        self.thread.progress_update.connect(self.signal_accept_pb)

        self.thread.question_signal.connect(self.show_popup)
        self.thread.setTerminationEnabled(True)
        self.thread.start()

    def startLearn(self):
        self.start_learn.setEnabled(False)
        self.take_osci.setEnabled(False)
        self.ml_learn = learnThread(self.path)
        # self.ml_learn.done_signal.connect(self.osci_status_label_now)
        self.ml_learn.setTerminationEnabled(True)
        self.ml_learn.done_signal.connect(self.learn_done_signal)
        self.ml_learn.start()
        self.ml_predict.stop()
        self.start_predict.setEnabled(False)


    def startPredict(self):
        if self.osci_pause:
            self.osci_pause = False
            self.ml_predict.is_pause = False
            print('startPredict', self.osci_pause)
            self.start_predict.setText('Работаем')
            if self.ml_predict.isRunning():
                print('tread_work')
            else:
                print('thread rerun')
                self.ml_predict.start() #!!!!!!!!!!!!111111
        else:
            self.osci_pause = True
            self.ml_predict.is_pause = True
            self.start_predict.setText('Пауза')

    def show_popup(self, i):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        msg.button(QMessageBox.Yes).setText('Да')
        msg.button(QMessageBox.Cancel).setText('ГАЛЯ, ОТМЕНА!')
        msg.setWindowTitle('Вопрос')
        msg.setText('Начинаем? \nКласс: ' + str(self.thread.current_class) + '\nПозиция: ' + str(i))
        # msg.buttonClicked.connect(self.popup_button)
        result = msg.exec_()
        if result == QtWidgets.QMessageBox.Yes:
            self.thread.osci.is_pause = False
            # self.thread.is_pause = False
        else:
            self.thread.stop()


    def signal_accept_pb(self, i):
        self.progressBar.setValue(i)

    def learn_done_signal(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.setWindowTitle('Уведомление')
        msg.setText('Перенастройка выполнена!')
        msg.setFont(QFont('ms shell dlg 2', 12))
        result = msg.exec_()
        self.take_osci.setEnabled(True)
        self.start_learn.setEnabled(True)

        self.init_ml_predict()
        self.start_predict.setEnabled(True)

    def class_signal(self, i):
        self.predict_c.setText(self.predict_rename(i))
        self.font_resize(self.predict_c, self.predict_r)

    def range_signal(self, i):
        self.predict_r.setText('Дальность: '+str(i))
        self.font_resize(self.predict_c, self.predict_r)

    def predict_rename(self, c):
        print(self.class_db.get(c))
        return self.class_db.get(c)

    def font_resize_ratio(self, qlabel):
        rect_width = qlabel.rect().width()
        rect_height = qlabel.rect().height()
        label_length = qlabel.fontMetrics().boundingRect(qlabel.text()).width()
        label_height = qlabel.fontMetrics().boundingRect(qlabel.text()).height()
        return rect_width, rect_height, label_length, label_height

    def font_resize(self, predict_c, predict_r):
        default_size = 12
        predict_c.setFont(QFont('ms shell dlg 2', default_size))
        predict_r.setFont(QFont('ms shell dlg 2', default_size))
        rect_width_c, rect_height_c, label_length_c, label_height_c = self.font_resize_ratio(predict_c)
        rect_width_r, rect_height_r, label_length_r, label_height_r = self.font_resize_ratio(predict_r)
        if label_length_c > label_length_r:
            ratio_c = rect_width_c // label_length_c
            ratio_r = rect_height_r // label_height_r
        elif label_length_c < label_length_r:
            ratio_c = rect_height_c // label_height_c
            ratio_r = rect_width_r // label_length_r
        else:
            ratio_c = rect_width_c // label_length_c
            ratio_r = ratio_c
        # print('ratio_c: ', ratio_c)
        # print('ratio_r: ', ratio_r)
        fix = 50
        predict_c.setFont(QFont('ms shell dlg 2', default_size * ratio_c - fix))
        predict_r.setFont(QFont('ms shell dlg 2', default_size * ratio_r - fix))

    def addRowTable(self):
        rowCount = self.tableWidget.rowCount()

        if rowCount == 0:
            self.tableWidget.setRowCount(1)
            self.tableWidget.setItem(0, 0, self.item_alignment(str(0)))
            self.tableWidget.setItem(0, 1, self.item_alignment(''))
            self.class_chooser.addItem(str(0))
            # item = QTableWidgetItem(values_list[row])
            # item.setTextAlignment(QtCore.Qt.AlignCenter)
        else:
            self.tableWidget.insertRow(rowCount)
            self.tableWidget.setItem(rowCount,0,self.item_alignment(str(rowCount)))
            self.tableWidget.setItem(rowCount, 1, self.item_alignment(''))
            self.class_chooser.addItem(str(rowCount))

    def removeRowTable(self):
        rowCount = self.tableWidget.rowCount()
        if rowCount > 0:
            self.tableWidget.removeRow(rowCount-1)
            self.class_chooser.removeItem(rowCount-1)

    def saveTable(self):
        rowCount = self.tableWidget.rowCount()
        #columnCount = self.tableWidget.columnCount()
        dict = {}

        for i in range(rowCount):
            item = self.tableWidget.item(i, 1)
            if item is None:
                dict[self.tableWidget.item(i, 0).text()] = ""
            else:
                dict[self.tableWidget.item(i,0).text()] = self.tableWidget.item(i,1).text()
        if os.path.isfile(self.path + '/class_db.npy'):
            os.remove(self.path + '/class_db.npy')
            np.save(self.path + '/class_db.npy', dict)
        else:
            np.save(self.path + '/class_db.npy', dict)


        self.load_data()

    def combobox_changed(self, value):
        item = self.tableWidget.item(value,1)
        if item is None:
            self.class_name_label.setText('Не определено имя класса')
        else:
            self.class_name_label.setText(item.text())

class ReadOnlyDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        return

class progressThread(QThread):
    progress_update = pyqtSignal(int)
    question_signal = pyqtSignal(int)
    def __init__(self, osci, current_class, bottom, up, steps):

        self.osci = osci
        self.current_class = current_class
        self.bottom = bottom
        self.up = up
        self.steps = steps
        # self.is_pause = True

        super(progressThread, self).__init__()

    def __del__(self):
        self.wait()

    def stop(self):
        self.terminate()

    def run(self):
        print('dorun')

        self.osci.run(self.progress_update,self.question_signal, self.current_class, self.bottom, self.up, self.steps)

class learnThread(QThread):
    done_signal = pyqtSignal(str)
    def __init__(self, path):
        self.path = path

        super(learnThread, self).__init__()

    def __del__(self):
        self.wait()

    def stop(self):
        self.terminate()

    def run(self):
        print('dorun')
        ML_inst.ml_learn(self.path)

        self.done_signal.emit('done')

class predictThread(QThread):
    class_signal = pyqtSignal(str)
    range_signal = pyqtSignal(str)
    is_pause = True
    def __init__(self, path, osci):
        self.path = path
        self.ml = ML_inst.ml_predict(self.path)
        self.osci = osci

        print('predict_th_start')
        super(predictThread, self).__init__()

    def __del__(self):
        self.wait()

    def stop(self):
        self.terminate()
        print('predict_th_stop')

    def run(self):
        print('dorun')

        # df = pd.read_csv(self.path + "\data.csv")
        # for i in range(1,10):
        #     X_test = df.loc[(df["0"] == 2) & (df["1"] == i), :].iloc[0, 2:1252]
        #     c, r = self.ml.get_predict(X_test)
        #     print(r)
        #     self.class_signal.emit(str(int(c)))
        #     self.range_signal.emit(str(int(r)))
        #     # time.sleep(1)
        # print('done')
        while True:
            if self.is_pause:
                print('pause')
                time.sleep(10)
            else:

                X_test = self.osci.predict_run()
                c, r = self.ml.get_predict(X_test)
                #     print(r)
                self.class_signal.emit(str(int(c)))
                self.range_signal.emit(str(int(r)))





        # X_test =  self.osci.predict_run()
        # c, r = self.ml.get_predict(X_test)
        # #     print(r)
        # self.class_signal.emit(str(int(c)))
        # self.range_signal.emit(str(int(r)))



# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())
try:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    mainWindow = Ui_MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
except:

    mainWindow.osci.disconnect()
    print('CRASH!')
