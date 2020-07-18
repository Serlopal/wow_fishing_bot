from PyQt5 import QtCore, QtGui, QtWidgets
import utils
import os
import json
import pyautogui
import numpy as np
import win32gui
from mss import mss
import time
import os
import sys
from PyQt5.QtWidgets import QApplication, QGridLayout, QTextEdit, QMainWindow, QGroupBox, \
	QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt5.QtCore import pyqtSignal, QThread, QRectF
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, PlotWidget

import keyboard
import mouse
from scipy.stats import linregress
from collections import deque

pg.setConfigOption('background', 'w')

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
pyautogui.PAUSE = 0.0001
pyautogui.FAILSAFE = False

curr_dir = os.path.dirname(os.path.abspath(__file__))


class WowFishingBot:

    def __init__(self, ui):
        self.sct = mss()
        self.UI = ui
        self.dead_UI = False

        self.frame = None
        self.bait_window = None
        self.tries = 0

    def print_to_log(self, s):
        self.UI.log_viewer.emitter.emit(s)

    def throw_bait(self):
        pyautogui.hotkey(self.fishing_hotkey)
        time.sleep(1)

    def jump(self):
        self.print_to_log('Jump!')
        pyautogui.hotkey(' ')
        time.sleep(1)

    def loot(self):
        n_items = 2
        x_loot_coord = self.loot_coords_x
        y_loot_coord = self.loot_coords_y
        loot_delta = self.loot_vertical_shift

        # first item
        time.sleep(0.5)
        pyautogui.moveTo(x=x_loot_coord, y=y_loot_coord, duration=0.5)
        pyautogui.click()

        # next items
        for i in range(n_items):
            pyautogui.moveRel(4, 0, duration=0.05)
            pyautogui.moveRel(-4, 0, duration=0.05)
            pyautogui.moveRel(0, loot_delta, duration=0.05)

            pyautogui.click()

    def watch_bait(self, bait_coords):
        # apply offset to baitcoords
        bait_coords[0] += self.bait_window_offset_right - self.bait_window_offset_left
        bait_coords[1] += self.bait_window_offset_bottom - self.bait_window_offset_top

        # capturing of the float window
        bait_window = {'top': int(bait_coords[1] - self.bait_window_height / 2),
                       'left': int(bait_coords[0] - self.bait_window_width / 2),
                       'width': self.bait_window_width,
                       'height': self.bait_window_height}

        slope_samples_number = int(self.slope_estimation_samples)
        score_buffer = deque(maxlen=slope_samples_number)
        slope_buffer = []
        std_scale = self.bait_sensibility # float(self.UI.bait_mov_sensibility_edit.edit.text())

        self.UI.log_viewer.emitter.emit("watching float...")

        fishing_time = 30
        mask_stats_time = 2
        score_stats_time = 2
        watch_bait_time = fishing_time - mask_stats_time - score_stats_time

        # ---- GET BAIT MASK ------
        acc_mask = np.ones_like(self.get_bait_mask(bait_window)).astype('int')
        t = time.time()
        while time.time() - t < mask_stats_time:
            current_bait = self.get_bait_mask(bait_window).astype('int')
            self.display_bait_mask(current_bait)
            acc_mask += current_bait

        # ---- ESTIMATE DISTRIBUTION OF NORMAL NO FISH SCORES ------
        t = time.time()
        while time.time() - t < score_stats_time:
            current_bait = self.get_bait_mask(bait_window).astype('int')
            self.display_bait_mask(current_bait)
            score = np.sum(np.divide(current_bait, acc_mask))
            score_buffer.append(score)
            if len(score_buffer) >= slope_samples_number:
                slope_buffer.append(np.abs(linregress(np.arange(slope_samples_number), score_buffer).slope))
        slope_threshold = np.mean(slope_buffer) + std_scale * np.std(slope_buffer)

        # ---- WATCH BAIT FOR FISH ------
        score_buffer.clear()
        while time.time() - t < watch_bait_time:
            bait_image = np.array(self.sct.grab(bait_window))
            current_bait = self.process_bait(bait_image)
            score = np.sum(np.divide(current_bait, acc_mask))
            score_buffer.append(score)

            if len(score_buffer) >= slope_samples_number:
                slope = np.abs(linregress(np.arange(slope_samples_number), score_buffer).slope)
                if slope > slope_threshold:
                    time.sleep(4)
                    pyautogui.rightClick()
                    self.UI.log_viewer.emitter.emit("tried to capture fish")
                    time.sleep(0.2)
                    self.UI.log_viewer.emitter.emit("looting the fish...")
                    self.loot()
                    break

                self.display_bait_mask(current_bait)
                self.display_trigger_signal([slope, slope_threshold])

    def scan_grid(self):
        if self.dead_UI:
            exit()
        self.UI.log_viewer.emitter.emit("Throwing bait...")
        self.throw_bait()

        grid_step = 20
        found = False
        bait_coords = None

        a = int(self.frame[1] + self.frame[3] * self.grid_top_padding*0.01)
        b = int(self.frame[1] + self.frame[3] * (1 - self.grid_bottom_padding*0.01))
        c = int(self.frame[0] + self.frame[2] * self.grid_left_padding*0.01)
        d = int(self.frame[0] + self.frame[2] * (1 - self.grid_right_padding*0.01))

        for j in range(a, b, grid_step):
            if found:
                break
            for i in range(c, d, grid_step):
                pre_cursor = win32gui.GetCursorInfo()[1]
                utils.move_mouse([i, j])
                # time.sleep(0.02)
                post_cursor = win32gui.GetCursorInfo()[1]
                if pre_cursor != post_cursor:
                    found = True
                    j += int(self.frame[2] / 100)
                    pyautogui.moveRel(0, int(self.frame[2] / 100), duration=0.05)

                    self.print_to_log("Found bait at coordinates {0} , {1}".format(i, j))

                    bait_coords = [i, j]
                    break
        if bait_coords is not None:
            self.watch_bait(bait_coords)

        self.tries += 1
        self.UI.tries_digital_counter.display(str(self.tries))
        self.jump()

    def set_wow_frame(self, frame):
        self.frame = frame
        # dimension of the window that will encapsulate the bait
        self.bait_window = int(frame[3] / 4)

    def get_bait_mask(self, w):
        return self.process_bait(np.array(self.sct.grab(w)))

    @staticmethod
    def process_bait(img):
        return utils.binarize_canny(img)

    def display_bait_mask(self, bait_mask):
        self.UI.binary_bait_view.emitter.emit(np.rot90(bait_mask * 255, k=3))

    def display_trigger_signal(self, signals):
        self.UI.slope_signal_viewer.emitter.emit(signals)

    @property
    def grid_travelling_speed(self):
        return float(self.UI.grid_travelling_speed_slider.value())

    @property
    def slope_estimation_samples(self):
        return float(self.UI.slope_estimation_samples_slider.value())

    @property
    def bait_sensibility(self):
        return float(self.UI.bait_sensibility_slider.value())

    @property
    def grid_top_padding(self):
        return float(self.UI.grid_top_padding_slider.value())

    @property
    def grid_bottom_padding(self):
        return float(self.UI.grid_bottom_padding_slider.value())

    @property
    def grid_left_padding(self):
        return float(self.UI.grid_left_padding_slider.value())

    @property
    def grid_right_padding(self):
        return float(self.UI.grid_right_padding_slider.value())

    @property
    def loot_coords_x(self):
        return float(self.UI.loot_Xcoords_input.text())

    @property
    def loot_coords_y(self):
        return float(self.UI.loot_Ycoords_input.text())

    @property
    def loot_vertical_shift(self):
        return float(self.UI.loot_vertical_shift.text())

    @property
    def fishing_hotkey(self):
        return self.UI.fishing_hotkey_edit.text()

    @property
    def bait_window_width(self):
        return self.UI.bait_window_width_slider.value()

    @property
    def bait_window_height(self):
        return self.UI.bait_window_height_slider.value()

    @property
    def bait_window_offset_top(self):
        return self.UI.bait_window_offset_top_spinbox.value()

    @property
    def bait_window_offset_bottom(self):
        return self.UI.bait_window_offset_bottom_spinbox.value()

    @property
    def bait_window_offset_right(self):
        return self.UI.bait_window_offset_right_spinbox.value()

    @property
    def bait_window_offset_left(self):
        return self.UI.bait_window_offset_left_spinbox.value()


class QImshow(pg.GraphicsLayoutWidget):
    emitter = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.emitter.connect(lambda x: self.update_figure(x))

        self._view = self.addViewBox()
        self._view.setAspectLocked(True)
        self._img = pg.ImageItem(border='w')
        self._view.addItem(self._img)
        self._view.setRange(QRectF(0, 0, 128, 128))

        self.lock = time.time()
        self.fps = 25
        self.freq = 1/self.fps

    def update_figure(self, frame):
        if time.time() - self.lock > self.freq:
            self.lock = time.time()

            self._img.setImage(frame)

            self.repaint()
        else:
            pass


class QSignalViewer(pg.PlotWidget):
    emitter = pyqtSignal(object)

    def __init__(self, num_signals, yrange=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # save number of signals
        self.nplots = num_signals
        # set number of samples to be displayed per signal at a time
        self.nsamples = 500
        # connect the signal to be emitted by the feeder to the slot of the plotWidget that will update the signals
        self.emitter.connect(lambda values: self.update(values))
        # buffer to store the data from all signals
        self.buff = np.zeros((self.nplots, self.nsamples))
        # limit range
        if yrange:
            self.setYRange(*yrange)
        # create curves for the signals
        self.curves = []
        for i in range(self.nplots):
            c = pg.PlotCurveItem(pen=pg.mkPen((i, self.nplots * 1.3), width=3))
            self.addItem(c)
            self.curves.append(c)

    def update(self, data):
        # update buffer
        self.buff = np.concatenate([self.buff[:, 1:], np.reshape(data, (-1, 1))], axis=1)
        # update plots
        for i in range(self.nplots):
            self.curves[i].setData(self.buff[i])

    def update_signals(self, values):
        self.emitter.emit(values)


class Log(QTextEdit):
    emitter = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitter.connect(lambda text: self.update_log(text))

    def update_log(self, text):
        self.append("\n" + text)
        self.repaint()


class QtCache:
    def __init__(self, cache_path, ui):
        self.cache_path = cache_path
        self.ui = ui
        self.cache = None

    def load(self):
        """
        This function loads the cache from disk and fills the last used data into the
        input widgets by object name
        """

        if not os.path.exists(self.cache_path):
            return
        else:
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)

            # get widgets from cache
            for w in self.cache["widgets"]:
                widget_name = w["widget_name"]
                ui_widget = getattr(self.ui, widget_name)
                widget_type = w["widget_type"]
                widget_content = w["widget_value"]
                filler = getattr(self, f"fill_{widget_type}")
                filler(ui_widget, widget_content)

    def save(self):
        curr_widget_data = {"widgets": []}

        for w in self.cache["widgets"]:
            widget_name = w["widget_name"]
            ui_widget = getattr(self.ui, widget_name)
            widget_type = w["widget_type"]
            widget_content = w["widget_value"]
            getter = getattr(self, f"get_{widget_type}")
            contents = getter(ui_widget)

            curr_widget_data["widgets"].append({"widget_name": widget_name,
                                                "widget_type": widget_type,
                                                "widget_value": contents})

        # write current data in UI to cache as a json
        with open(self.cache_path, "w") as f:
            json.dump(curr_widget_data, f)

    def fill_QLineEdit(self, edit, text):
        edit.setText(text)

    def get_QLineEdit(self, edit):
        return edit.text()

    def fill_QSlider(self, slider, value):
        slider.setValue(value)

    def get_QSlider(self, slider):
        return slider.value()

    def fill_QSpinBox(self, slider, value):
        slider.setValue(value)

    def get_QSpinBox(self, slider):
        return slider.value()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1054, 813)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.start_fishing_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_fishing_button.setEnabled(False)
        self.start_fishing_button.setGeometry(QtCore.QRect(10, 20, 281, 91))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.start_fishing_button.setFont(font)
        self.start_fishing_button.setObjectName("start_fishing_button")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 190, 301, 171))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tries_digital_counter = QtWidgets.QLCDNumber(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(False)
        font.setWeight(50)
        self.tries_digital_counter.setFont(font)
        self.tries_digital_counter.setObjectName("tries_digital_counter")
        self.horizontalLayout.addWidget(self.tries_digital_counter)
        self.stop_fishing_label = QtWidgets.QLabel(self.centralwidget)
        self.stop_fishing_label.setEnabled(True)
        self.stop_fishing_label.setGeometry(QtCore.QRect(10, 130, 301, 71))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.stop_fishing_label.setFont(font)
        self.stop_fishing_label.setObjectName("stop_fishing_label")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 370, 1011, 381))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.find_wow_tab = QtWidgets.QWidget()
        self.find_wow_tab.setObjectName("find_wow_tab")
        self.formLayoutWidget = QtWidgets.QWidget(self.find_wow_tab)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 20, 281, 71))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.wow_process_name_input = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.wow_process_name_input.setObjectName("wow_process_name_input")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.wow_process_name_input)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.wow_window_name_input = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.wow_window_name_input.setObjectName("wow_window_name_input")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.wow_window_name_input)
        self.wow_found_label = QtWidgets.QLabel(self.find_wow_tab)
        self.wow_found_label.setGeometry(QtCore.QRect(20, 160, 901, 61))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.wow_found_label.setFont(font)
        self.wow_found_label.setObjectName("wow_found_label")
        self.find_wow_button = QtWidgets.QPushButton(self.find_wow_tab)
        self.find_wow_button.setGeometry(QtCore.QRect(20, 100, 93, 28))
        self.find_wow_button.setObjectName("find_wow_button")
        self.tabWidget.addTab(self.find_wow_tab, "")
        self.find_bait_tab = QtWidgets.QWidget()
        self.find_bait_tab.setObjectName("find_bait_tab")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.find_bait_tab)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(280, 10, 711, 331))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label99_2 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label99_2.setObjectName("label99_2")
        self.gridLayout_3.addWidget(self.label99_2, 4, 6, 1, 1)
        self.label99_3 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label99_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label99_3.setObjectName("label99_3")
        self.gridLayout_3.addWidget(self.label99_3, 7, 3, 1, 1)
        self.grid_travelling_speed_slider_label_2 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.grid_travelling_speed_slider_label_2.setMinimumSize(QtCore.QSize(47, 0))
        self.grid_travelling_speed_slider_label_2.setMaximumSize(QtCore.QSize(47, 16777215))
        self.grid_travelling_speed_slider_label_2.setLineWidth(2)
        self.grid_travelling_speed_slider_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_travelling_speed_slider_label_2.setIndent(0)
        self.grid_travelling_speed_slider_label_2.setObjectName("grid_travelling_speed_slider_label_2")
        self.gridLayout_3.addWidget(self.grid_travelling_speed_slider_label_2, 4, 2, 1, 1)
        self.grid_right_padding_slider = QtWidgets.QSlider(self.gridLayoutWidget_3)
        self.grid_right_padding_slider.setMaximum(50)
        self.grid_right_padding_slider.setOrientation(QtCore.Qt.Horizontal)
        self.grid_right_padding_slider.setInvertedAppearance(True)
        self.grid_right_padding_slider.setObjectName("grid_right_padding_slider")
        self.gridLayout_3.addWidget(self.grid_right_padding_slider, 4, 5, 1, 1)
        self.label99 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label99.setObjectName("label99")
        self.gridLayout_3.addWidget(self.label99, 4, 0, 1, 1)
        self.label99_4 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label99_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label99_4.setObjectName("label99_4")
        self.gridLayout_3.addWidget(self.label99_4, 1, 3, 1, 1)
        self.grid_travelling_speed_slider_label_5 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.grid_travelling_speed_slider_label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_travelling_speed_slider_label_5.setObjectName("grid_travelling_speed_slider_label_5")
        self.gridLayout_3.addWidget(self.grid_travelling_speed_slider_label_5, 4, 3, 1, 1)
        self.grid_travelling_speed_slider_label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.grid_travelling_speed_slider_label.setMinimumSize(QtCore.QSize(47, 0))
        self.grid_travelling_speed_slider_label.setMaximumSize(QtCore.QSize(47, 56))
        self.grid_travelling_speed_slider_label.setLineWidth(2)
        self.grid_travelling_speed_slider_label.setMidLineWidth(0)
        self.grid_travelling_speed_slider_label.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_travelling_speed_slider_label.setWordWrap(False)
        self.grid_travelling_speed_slider_label.setIndent(0)
        self.grid_travelling_speed_slider_label.setObjectName("grid_travelling_speed_slider_label")
        self.gridLayout_3.addWidget(self.grid_travelling_speed_slider_label, 4, 4, 1, 1)
        self.grid_travelling_speed_slider_label_4 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.grid_travelling_speed_slider_label_4.setLineWidth(2)
        self.grid_travelling_speed_slider_label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_travelling_speed_slider_label_4.setObjectName("grid_travelling_speed_slider_label_4")
        self.gridLayout_3.addWidget(self.grid_travelling_speed_slider_label_4, 5, 3, 1, 1)
        self.grid_bottom_padding_slider = QtWidgets.QSlider(self.gridLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grid_bottom_padding_slider.sizePolicy().hasHeightForWidth())
        self.grid_bottom_padding_slider.setSizePolicy(sizePolicy)
        self.grid_bottom_padding_slider.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.grid_bottom_padding_slider.setMaximum(50)
        self.grid_bottom_padding_slider.setOrientation(QtCore.Qt.Vertical)
        self.grid_bottom_padding_slider.setObjectName("grid_bottom_padding_slider")
        self.gridLayout_3.addWidget(self.grid_bottom_padding_slider, 6, 3, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.grid_left_padding_slider = QtWidgets.QSlider(self.gridLayoutWidget_3)
        self.grid_left_padding_slider.setMaximum(50)
        self.grid_left_padding_slider.setSingleStep(1)
        self.grid_left_padding_slider.setOrientation(QtCore.Qt.Horizontal)
        self.grid_left_padding_slider.setObjectName("grid_left_padding_slider")
        self.gridLayout_3.addWidget(self.grid_left_padding_slider, 4, 1, 1, 1)
        self.grid_travelling_speed_slider_label_3 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.grid_travelling_speed_slider_label_3.setLineWidth(2)
        self.grid_travelling_speed_slider_label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_travelling_speed_slider_label_3.setObjectName("grid_travelling_speed_slider_label_3")
        self.gridLayout_3.addWidget(self.grid_travelling_speed_slider_label_3, 3, 3, 1, 1)
        self.grid_top_padding_slider = QtWidgets.QSlider(self.gridLayoutWidget_3)
        self.grid_top_padding_slider.setMaximum(50)
        self.grid_top_padding_slider.setOrientation(QtCore.Qt.Vertical)
        self.grid_top_padding_slider.setInvertedAppearance(True)
        self.grid_top_padding_slider.setObjectName("grid_top_padding_slider")
        self.gridLayout_3.addWidget(self.grid_top_padding_slider, 2, 3, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.gridLayoutWidget = QtWidgets.QWidget(self.find_bait_tab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 100, 221, 131))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label99_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label99_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label99_6.setObjectName("label99_6")
        self.gridLayout.addWidget(self.label99_6, 1, 2, 1, 1)
        self.label99_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label99_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label99_5.setObjectName("label99_5")
        self.gridLayout.addWidget(self.label99_5, 0, 0, 1, 2)
        self.label99_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label99_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label99_7.setObjectName("label99_7")
        self.gridLayout.addWidget(self.label99_7, 2, 1, 1, 1)
        self.grid_travelling_speed_slider = QtWidgets.QSlider(self.gridLayoutWidget)
        self.grid_travelling_speed_slider.setOrientation(QtCore.Qt.Horizontal)
        self.grid_travelling_speed_slider.setObjectName("grid_travelling_speed_slider")
        self.gridLayout.addWidget(self.grid_travelling_speed_slider, 1, 1, 1, 1)
        self.grid_travelling_speed_slider_2 = QtWidgets.QSlider(self.gridLayoutWidget)
        self.grid_travelling_speed_slider_2.setOrientation(QtCore.Qt.Horizontal)
        self.grid_travelling_speed_slider_2.setObjectName("grid_travelling_speed_slider_2")
        self.gridLayout.addWidget(self.grid_travelling_speed_slider_2, 3, 1, 1, 1)
        self.label99_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label99_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label99_8.setObjectName("label99_8")
        self.gridLayout.addWidget(self.label99_8, 3, 2, 1, 1)
        self.tabWidget.addTab(self.find_bait_tab, "")
        self.watch_bait_tab = QtWidgets.QWidget()
        self.watch_bait_tab.setObjectName("watch_bait_tab")
        self.binary_bait_view = QImshow(self.watch_bait_tab)  # GraphicsLayoutWidget(self.watch_bait_tab)
        self.binary_bait_view.setGeometry(QtCore.QRect(320, 10, 331, 251))
        self.binary_bait_view.setObjectName("binary_bait_view")
        self.slope_signal_viewer = QSignalViewer(2, None, self.watch_bait_tab)  # PlotWidget(self.watch_bait_tab)
        self.slope_signal_viewer.setGeometry(QtCore.QRect(660, 10, 341, 251))
        self.slope_signal_viewer.setObjectName("slope_signal_viewer")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.watch_bait_tab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 301, 191))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.bait_window_width_slider = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.bait_window_width_slider.setMaximum(300)
        self.bait_window_width_slider.setOrientation(QtCore.Qt.Horizontal)
        self.bait_window_width_slider.setObjectName("bait_window_width_slider")
        self.gridLayout_2.addWidget(self.bait_window_width_slider, 5, 0, 1, 1)
        self.bait_sensibility_slider = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.bait_sensibility_slider.setOrientation(QtCore.Qt.Horizontal)
        self.bait_sensibility_slider.setObjectName("bait_sensibility_slider")
        self.gridLayout_2.addWidget(self.bait_sensibility_slider, 3, 0, 1, 1)
        self.slope_estimation_sample_slider_label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.slope_estimation_sample_slider_label.setObjectName("slope_estimation_sample_slider_label")
        self.gridLayout_2.addWidget(self.slope_estimation_sample_slider_label, 1, 1, 1, 1)
        self.bait_sensiblity_slider_2 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.bait_sensiblity_slider_2.setObjectName("bait_sensiblity_slider_2")
        self.gridLayout_2.addWidget(self.bait_sensiblity_slider_2, 3, 1, 1, 1)
        self.slope_estimation_samples_slider = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.slope_estimation_samples_slider.setOrientation(QtCore.Qt.Horizontal)
        self.slope_estimation_samples_slider.setObjectName("slope_estimation_samples_slider")
        self.gridLayout_2.addWidget(self.slope_estimation_samples_slider, 1, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 4, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 6, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 2, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 0, 0, 1, 1)
        self.bait_sensiblity_slider_3 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.bait_sensiblity_slider_3.setObjectName("bait_sensiblity_slider_3")
        self.gridLayout_2.addWidget(self.bait_sensiblity_slider_3, 5, 1, 1, 1)
        self.bait_window_height_slider = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.bait_window_height_slider.setMaximum(300)
        self.bait_window_height_slider.setOrientation(QtCore.Qt.Horizontal)
        self.bait_window_height_slider.setObjectName("bait_window_height_slider")
        self.gridLayout_2.addWidget(self.bait_window_height_slider, 7, 0, 1, 1)
        self.bait_sensiblity_slider_4 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.bait_sensiblity_slider_4.setObjectName("bait_sensiblity_slider_4")
        self.gridLayout_2.addWidget(self.bait_sensiblity_slider_4, 7, 1, 1, 1)
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.watch_bait_tab)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 240, 178, 97))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.bait_window_offset_left_spinbox = QtWidgets.QSpinBox(self.gridLayoutWidget_4)
        self.bait_window_offset_left_spinbox.setMaximum(300)
        self.bait_window_offset_left_spinbox.setObjectName("bait_window_offset_left_spinbox")
        self.gridLayout_4.addWidget(self.bait_window_offset_left_spinbox, 1, 0, 1, 1)
        self.bait_window_offset_top_spinbox = QtWidgets.QSpinBox(self.gridLayoutWidget_4)
        self.bait_window_offset_top_spinbox.setMaximum(300)
        self.bait_window_offset_top_spinbox.setObjectName("bait_window_offset_top_spinbox")
        self.gridLayout_4.addWidget(self.bait_window_offset_top_spinbox, 0, 1, 1, 1)
        self.bait_window_offset_right_spinbox = QtWidgets.QSpinBox(self.gridLayoutWidget_4)
        self.bait_window_offset_right_spinbox.setMaximum(300)
        self.bait_window_offset_right_spinbox.setObjectName("bait_window_offset_right_spinbox")
        self.gridLayout_4.addWidget(self.bait_window_offset_right_spinbox, 1, 2, 1, 1)
        self.bait_window_offset_bottom_spinbox = QtWidgets.QSpinBox(self.gridLayoutWidget_4)
        self.bait_window_offset_bottom_spinbox.setMaximum(300)
        self.bait_window_offset_bottom_spinbox.setObjectName("bait_window_offset_bottom_spinbox")
        self.gridLayout_4.addWidget(self.bait_window_offset_bottom_spinbox, 2, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.watch_bait_tab)
        self.label_13.setGeometry(QtCore.QRect(10, 200, 137, 37))
        self.label_13.setObjectName("label_13")
        self.tabWidget.addTab(self.watch_bait_tab, "")
        self.loot_fish_tab = QtWidgets.QWidget()
        self.loot_fish_tab.setObjectName("loot_fish_tab")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.loot_fish_tab)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(20, 20, 281, 261))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_6.setObjectName("label_6")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.loot_Xcoords_input = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.loot_Xcoords_input.setObjectName("loot_Xcoords_input")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.loot_Xcoords_input)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_7.setObjectName("label_7")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.loot_Ycoords_input = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.loot_Ycoords_input.setObjectName("loot_Ycoords_input")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.loot_Ycoords_input)
        self.label_10 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_10.setObjectName("label_10")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.loot_vertical_shift = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.loot_vertical_shift.setObjectName("loot_vertical_shift")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.loot_vertical_shift)
        self.record_loot_coords_button = QtWidgets.QPushButton(self.loot_fish_tab)
        self.record_loot_coords_button.setGeometry(QtCore.QRect(350, 20, 221, 28))
        self.record_loot_coords_button.setObjectName("record_loot_coords_button")
        self.tabWidget.addTab(self.loot_fish_tab, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(20, 20, 257, 80))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.fishing_wait_time_edit = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.fishing_wait_time_edit.setObjectName("fishing_wait_time_edit")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fishing_wait_time_edit)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.fishing_hotkey_edit = QtWidgets.QLineEdit(self.formLayoutWidget_2)
        self.fishing_hotkey_edit.setObjectName("fishing_hotkey_edit")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.fishing_hotkey_edit)
        self.tabWidget.addTab(self.tab, "")
        self.log_viewer = Log(self.centralwidget)  # QtWidgets.QPlainTextEdit(self.centralwidget)
        self.log_viewer.setGeometry(QtCore.QRect(320, 20, 711, 341))
        self.log_viewer.setObjectName("log_viewer")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1054, 26))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionFind_WOW = QtWidgets.QAction(MainWindow)
        self.actionFind_WOW.setObjectName("actionFind_WOW")
        self.actionFind_bait = QtWidgets.QAction(MainWindow)
        self.actionFind_bait.setObjectName("actionFind_bait")
        self.actionWatch_bait = QtWidgets.QAction(MainWindow)
        self.actionWatch_bait.setObjectName("actionWatch_bait")
        self.actionLoot_fish = QtWidgets.QAction(MainWindow)
        self.actionLoot_fish.setObjectName("actionLoot_fish")
        self.menubar.addAction(self.menuMenu.menuAction())
        self.label_3.setBuddy(self.wow_process_name_input)
        self.label_4.setBuddy(self.wow_window_name_input)
        self.label99_2.setBuddy(self.grid_left_padding_slider)
        self.label99_3.setBuddy(self.grid_left_padding_slider)
        self.grid_travelling_speed_slider_label_2.setBuddy(self.grid_left_padding_slider)
        self.label99.setBuddy(self.grid_left_padding_slider)
        self.label99_4.setBuddy(self.grid_left_padding_slider)
        self.grid_travelling_speed_slider_label_5.setBuddy(self.grid_left_padding_slider)
        self.grid_travelling_speed_slider_label.setBuddy(self.grid_left_padding_slider)
        self.grid_travelling_speed_slider_label_4.setBuddy(self.grid_left_padding_slider)
        self.grid_travelling_speed_slider_label_3.setBuddy(self.grid_left_padding_slider)
        self.label99_6.setBuddy(self.grid_left_padding_slider)
        self.label99_5.setBuddy(self.grid_left_padding_slider)
        self.label99_7.setBuddy(self.grid_left_padding_slider)
        self.label99_8.setBuddy(self.grid_left_padding_slider)
        self.label_6.setBuddy(self.loot_Xcoords_input)
        self.label_7.setBuddy(self.loot_Ycoords_input)
        self.label_10.setBuddy(self.loot_Ycoords_input)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        self.grid_left_padding_slider.valueChanged['int'].connect(self.grid_travelling_speed_slider_label_2.setNum)
        self.grid_right_padding_slider.valueChanged['int'].connect(self.grid_travelling_speed_slider_label.setNum)
        self.grid_bottom_padding_slider.valueChanged['int'].connect(self.grid_travelling_speed_slider_label_4.setNum)
        self.grid_top_padding_slider.valueChanged['int'].connect(self.grid_travelling_speed_slider_label_3.setNum)
        self.slope_estimation_samples_slider.valueChanged['int'].connect(self.slope_estimation_sample_slider_label.setNum)
        self.bait_sensibility_slider.valueChanged['int'].connect(self.bait_sensiblity_slider_2.setNum)
        self.grid_travelling_speed_slider.valueChanged['int'].connect(self.label99_6.setNum)
        self.grid_travelling_speed_slider_2.valueChanged['int'].connect(self.label99_8.setNum)
        self.bait_window_width_slider.valueChanged['int'].connect(self.bait_sensiblity_slider_3.setNum)
        self.bait_window_height_slider.valueChanged['int'].connect(self.bait_sensiblity_slider_4.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.start_fishing_button.setText(_translate("MainWindow", "Start fishing"))
        self.stop_fishing_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:20pt; color:#ff0000;\">Jump to stop fishing</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "WOW process name"))
        self.label_4.setText(_translate("MainWindow", "WOW window name"))
        self.wow_found_label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" color:#ff0000;\">WOW not found yet</span></p><p><br/></p></body></html>"))
        self.find_wow_button.setText(_translate("MainWindow", "Find WOW"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.find_wow_tab), _translate("MainWindow", "Find WOW"))
        self.label99_2.setText(_translate("MainWindow", "right grid padding"))
        self.label99_3.setText(_translate("MainWindow", "bottom grid padding"))
        self.grid_travelling_speed_slider_label_2.setText(_translate("MainWindow", "0"))
        self.label99.setText(_translate("MainWindow", "left grid padding"))
        self.label99_4.setText(_translate("MainWindow", "top grid padding"))
        self.grid_travelling_speed_slider_label_5.setText(_translate("MainWindow", "+"))
        self.grid_travelling_speed_slider_label.setText(_translate("MainWindow", "0"))
        self.grid_travelling_speed_slider_label_4.setText(_translate("MainWindow", "0"))
        self.grid_travelling_speed_slider_label_3.setText(_translate("MainWindow", "0"))
        self.label99_6.setText(_translate("MainWindow", "0"))
        self.label99_5.setText(_translate("MainWindow", "grid travelling speed"))
        self.label99_7.setText(_translate("MainWindow", "grid step"))
        self.label99_8.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.find_bait_tab), _translate("MainWindow", "Find bait"))
        self.slope_estimation_sample_slider_label.setText(_translate("MainWindow", "0"))
        self.bait_sensiblity_slider_2.setText(_translate("MainWindow", "0"))
        self.label_11.setText(_translate("MainWindow", "Bait window width"))
        self.label_12.setText(_translate("MainWindow", "Bait window height"))
        self.label_9.setText(_translate("MainWindow", "Bait movement sensibility"))
        self.label_8.setText(_translate("MainWindow", "Slope estimation samples"))
        self.bait_sensiblity_slider_3.setText(_translate("MainWindow", "0"))
        self.bait_sensiblity_slider_4.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "+"))
        self.label_13.setText(_translate("MainWindow", "bait window offset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.watch_bait_tab), _translate("MainWindow", "Watch bait"))
        self.label_6.setText(_translate("MainWindow", "Top looting coordinates X"))
        self.label_7.setText(_translate("MainWindow", "Top looting coordinates Y"))
        self.label_10.setText(_translate("MainWindow", "Vertical looting shift"))
        self.record_loot_coords_button.setText(_translate("MainWindow", "Record looting coordinates"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.loot_fish_tab), _translate("MainWindow", "Loot fish"))
        self.label.setText(_translate("MainWindow", "wait time before fishing"))
        self.label_2.setText(_translate("MainWindow", "Fishing hot key"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Other settings"))
        self.menuMenu.setTitle(_translate("MainWindow", "Configure"))
        self.actionFind_WOW.setText(_translate("MainWindow", "Find WOW"))
        self.actionFind_bait.setText(_translate("MainWindow", "Find bait"))
        self.actionWatch_bait.setText(_translate("MainWindow", "Watch bait"))
        self.actionLoot_fish.setText(_translate("MainWindow", "Loot fish"))

    # **********************************
    def __init__(self):
        self.cache = QtCache(os.path.join(curr_dir, "cache.json"), self)
        self.fishing_lock = False

    def connect_cables(self):
        # button that attempts to locate WOW running
        self.find_wow_button.clicked.connect(self.find_wow)
        # hide label that reminds the user how to stop auto fishing
        self.stop_fishing_label.setVisible(False)
        # button that start the fishing process
        self.start_fishing_button.clicked.connect(self.start_fishing)
        # connect button to record coordinates of looting
        self.record_loot_coords_button.clicked.connect(self.record_looting_cords)

    def find_wow(self):
        # get window and process names from input
        process_name = self.wow_process_name_input.text()
        window_name = self.wow_window_name_input.text()

        if not process_name or not window_name:
            return

        # check process is running and we can find the window
        process_running = utils.check_process(process_name)
        window = utils.get_window(window_name)

        # check Wow is running
        if process_running and window:
            bot.set_wow_frame(window)
            self.wow_found_label.setText("WOW FOUND")
            self.start_fishing_button.setStyleSheet("background: green;")
            self.start_fishing_button.setEnabled(True)

    def start_fishing(self):
        if self.fishing_lock:
            return
        self.fishing_lock = True
        # activate warning on how to stop fishing
        self.stop_fishing_label.setVisible(True)

        # launch fishing thread in parallel
        self.fishing_thread = QThread()
        self.fishing_thread.run = self._fish
        self.fishing_thread.start()

        # watch if the user jumps to stop fishing
        while True:
            time.sleep(0.001)
            app.processEvents()
            if keyboard.is_pressed(" "):
                # kill fishing thread
                self.fishing_thread.terminate()
                # remove warning on how to sto fishing
                self.stop_fishing_label.setVisible(False)
                self.fishing_lock = False
                return

    def _fish(self):
        # give user time to place mouse on WOW's window
        self.log_viewer.emitter.emit("Place the cursor inside the WOW window")
        for i in reversed(range(int(self.fishing_wait_time_edit.text()))):
            time.sleep(1)
            self.log_viewer.emitter.emit("Fishing will start in {} ...".format(i))
            app.processEvents()

        while True:
            bot.scan_grid()

    def record_looting_cords(self):
        while True:
            if mouse.is_pressed("left"):
                x, y = pyautogui.position()
                self.loot_Xcoords_input.setText(str(x))
                self.loot_Ycoords_input.setText(str(y))
                break

    def load_cache(self):
        self.cache.load()

    def save_cache(self, _):
        self.cache.save()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    # create UI widgets
    ui.setupUi(MainWindow)
    # create callbacks
    ui.connect_cables()
    # load cache frrom disk
    ui.load_cache()
    # register cache saving at close time
    MainWindow.closeEvent = ui.save_cache
    # pass UI to bot
    bot = WowFishingBot(ui)

    MainWindow.show()
    sys.exit(app.exec_())

