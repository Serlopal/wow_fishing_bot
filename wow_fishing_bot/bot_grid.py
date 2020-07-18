import pyautogui
import numpy as np
import win32gui
from mss import mss
import time
import utils
import os
from PyQt5.QtWidgets import QApplication, QGridLayout, QTextEdit, QMainWindow, QGroupBox, \
	QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt5.QtCore import pyqtSignal, QThread, QRectF
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import keyboard
from scipy.stats import linregress
from collections import deque

pg.setConfigOption('background', 'w')

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
pyautogui.PAUSE = 0.0001
pyautogui.FAILSAFE = False


def monotonically_increasing(a):
	return np.all(a[1:] >= a[:-1], axis=0)


class Log(QTextEdit):
	emitter = pyqtSignal(object)

	def __init__(self):
		super().__init__()
		self.emitter.connect(lambda text: self.update_log(text))

	def update_log(self, text):
		self.append("\n" + text)
		self.repaint()


class LabeledLineEdit(QGroupBox):
	visibility_emitter = pyqtSignal(bool)

	def __init__(self, label, default_text):
		super().__init__()
		layout = QHBoxLayout()
		edit_label = QLabel(label)
		layout.addWidget(edit_label)
		self.edit = QLineEdit(default_text)
		layout.addWidget(self.edit)
		self.setLayout(layout)

		self.setFlat(True)
		self.setStyleSheet("border:0;")

	def set_visibility(self, v):
		self.setVisible(v)
		self.repaint()


class DynamicLabel(QLabel):
	text_emitter = pyqtSignal(str)
	visibility_emitter = pyqtSignal(bool)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.text_emitter.connect(lambda t: self.update_text(t))
		self.visibility_emitter.connect(lambda v: self.set_visibility(v))

	def update_text(self, text):
		self.setText(text)
		self.repaint()

	def set_visibility(self, v):
		self.setVisible(v)
		self.repaint()


class QImshow(pg.GraphicsLayoutWidget):
	emitter = pyqtSignal(object)

	def __init__(self):
		super().__init__()

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

	def __init__(self, num_signals, yrange=None):
		super().__init__()
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

# ----------------------------------------


class WowFishingBotUI:

	def __init__(self):
		self.startStop_fishing_flag = False
		self.app = None
		self.window = None

		self.game_process_name = None
		self.window_name = None

		self.log_viewer = None
		self.auto_fish_toggle = None
		self.find_wow_button = None
		self.fish_button = None
		self.fish_key_edit = None
		self.fishing_wait_time_edit = None
		self.stop_fishing_label = None
		self.fishing_thread = None
		self.tries_count_label = None
		self.loot_coords_edit = None
		self.loot_delta_edit = None
		self.bait_mov_sensibility_edit = None
		self.binary_image_widget = None
		self.rgb_image_widget = None
		self.background_model = None
		self.score_signal_viewer = None
		self.slope_signal_viewer = None
		self.post_detection_sleep_edit = None
		self.slope_samples_edit = None

		self.create_ui()

		self.bot = WowFishingBot(self)

		self.start_ui()

	def create_ui(self):
		# create QT application and main window
		self.app = QApplication(["WowFishingBotUI"])
		self.window = QMainWindow()
		self.window.setGeometry(1100, 50, 800, 900)
		self.window.setWindowTitle("WOW fishing bot")

		# create central widget and layout
		central_widget = QGroupBox()
		layout = QGridLayout()

		# CONTROLS AND TOGGLES
		# edit to change games process name
		self.game_process_name = LabeledLineEdit("WOW process name: ", "Wow.exe")
		layout.addWidget(self.game_process_name, 0, 0, 1, 1)
		# edit to change games window name
		self.window_name = LabeledLineEdit("WOW window name: ", "World of Warcraft")
		layout.addWidget(self.window_name, 0, 1, 1, 1)

		# fishing key
		self.fish_key_edit = LabeledLineEdit("Fishing key:", "0")
		layout.addWidget(self.fish_key_edit, 2, 0, 1, 1)

		# fishing wait time
		self.fishing_wait_time_edit = LabeledLineEdit("Fishing start delay", "4")
		layout.addWidget(self.fishing_wait_time_edit, 2, 1, 1, 1)

		# looting coordinates
		self.loot_coords_edit = LabeledLineEdit("Looting coords", "0.048 0.31")
		layout.addWidget(self.loot_coords_edit, 3, 0, 1, 1)

		# looting delta
		self.loot_delta_edit = LabeledLineEdit("Vertical looting delta", "0.03")
		layout.addWidget(self.loot_delta_edit, 3, 1, 1, 1)

		# bait movement sensibility (in standard deviations)
		self.bait_mov_sensibility_edit = LabeledLineEdit("Bait movement sensibility", "4")
		layout.addWidget(self.bait_mov_sensibility_edit, 4, 0, 1, 1)

		# bait movement sensibility (in standard deviations)
		self.post_detection_sleep_edit = LabeledLineEdit("Post-detection sleep time", "0.0")
		layout.addWidget(self.post_detection_sleep_edit, 4, 1, 1, 1)

		# bait movement sensibility (in standard deviations)
		self.slope_samples_edit = LabeledLineEdit("Slope estimation samples", "30")
		layout.addWidget(self.slope_samples_edit, 5, 0, 1, 1)

		# Fish! button
		self.fish_button = QPushButton("Fish")
		self.fish_button.clicked.connect(self.start_fishing)
		self.fish_button.setEnabled(False)
		layout.addWidget(self.fish_button, 6, 1, 1, 1)

		# button to try to find the process from the line edit
		self.find_wow_button = QPushButton("Find WOW!")
		self.find_wow_button.clicked.connect(self.find_wow)
		layout.addWidget(self.find_wow_button, 6, 0, 1, 1)

		# toggle for loop fishing
		self.auto_fish_toggle = QCheckBox("Auto pilot")
		self.auto_fish_toggle.setChecked(True)
		layout.addWidget(self.auto_fish_toggle, 7, 0, 1, 1)

		# warning label with hotkey to stop fishing
		self.stop_fishing_label = DynamicLabel("Jump to stop fishing")
		self.stop_fishing_label.setFont(QFont("Times", 12, QFont.Bold))
		self.stop_fishing_label.setStyleSheet("color: red;")
		self.stop_fishing_label.setVisible(False)
		layout.addWidget(self.stop_fishing_label, 8, 0, 1, 3)

		# label with the number of captures
		self.tries_count_label = DynamicLabel("0 tries")
		self.tries_count_label.setFont(QFont("Times", 24, QFont.Bold))
		self.tries_count_label.setStyleSheet("color: red;")
		layout.addWidget(self.tries_count_label, 9, 0, 1, 3)

		# LOG FROM BOT ACTIVITY
		self.log_viewer = Log()
		layout.addWidget(self.log_viewer, 12, 0, 3, 2)

		# image display
		self.binary_image_widget = QImshow()
		layout.addWidget(self.binary_image_widget, 15, 0, 5, 1)

		self.slope_signal_viewer = QSignalViewer(2, None)
		layout.addWidget(self.slope_signal_viewer, 15, 1, 5, 1)

		central_widget.setLayout(layout)
		self.window.setCentralWidget(central_widget)

		# register flag callback to let the backend know the UI has died
		self.app.aboutToQuit.connect(self.kill_bot)

	def start_ui(self):
		self.window.show()
		self.app.exec_()

	def _fish(self):
		while True:
			self.bot.fish_grid()
			if not self.auto_fish_toggle.isChecked():
				break

	def start_fishing(self):
		# activate warning on how to stop fishing
		self.stop_fishing_label.visibility_emitter.emit(True)

		# give user time to place mouse on WOW's window
		self.log_viewer.emitter.emit("Place the cursor inside the WOW window")
		for i in reversed(range(int(self.fishing_wait_time_edit.edit.text()))):
			time.sleep(1)
			self.log_viewer.emitter.emit("Fishing will start in {} ...".format(i))

		# launch fishing thread in parallel
		self.fishing_thread = QThread()
		self.fishing_thread.run = self._fish
		self.fishing_thread.start()

		# watch if the user jumps to stop fishing
		while True:
			time.sleep(0.001)
			self.app.processEvents()
			if keyboard.is_pressed(" "):
				# kill fishing thread
				self.fishing_thread.terminate()
				# remove warning on how to sto fishing
				self.stop_fishing_label.visibility_emitter.emit(False)

				return

	def find_wow(self):

		process_running = utils.check_process(self.game_process_name.edit.text())
		window = utils.get_window(self.window_name.edit.text())

		# check Wow is running
		if process_running and window:
			self.bot.set_wow_frame(window)
			self.log_viewer.emitter.emit("Wow window at" + str(self.bot.frame))
			self.fish_button.setEnabled(True)
			self.find_wow_button.setStyleSheet("background: green;")
		else:
			self.log_viewer.emitter.emit("Wow not found running")
			self.fish_button.setEnabled(False)
			self.find_wow_button.setStyleSheet("")

	def kill_bot(self):
		self.bot.dead_UI = True


class WowFishingBot:

	def __init__(self, ui):
		self.sct = mss()
		self.UI = ui
		self.dead_UI = False
		self.grid_frac_hor = [0.4, 0.6]
		self.grid_frac_ver = [0.1, 0.6]
		self.frame = None
		self.bait_window = None
		self.slope_samples = 3
		self.tries = 0

	def throw_bait(self):
		pyautogui.hotkey(self.UI.fish_key_edit.edit.text())
		time.sleep(1)

	def jump(self):
		self.UI.log_viewer.emitter.emit('Jump!')
		pyautogui.hotkey(' ')
		time.sleep(1)

	def loot(self):
		nitems = 3
		loot_coords = [float(x) for x in self.UI.loot_coords_edit.edit.text().split(" ")]
		loot_delta = float(self.UI.loot_delta_edit.edit.text())
		for i in range(nitems):
			time.sleep(0.5)
			pyautogui.moveTo(x=self.frame[0] + (self.frame[2] - self.frame[0]) * loot_coords[0],
							 y=self.frame[1] + (self.frame[3] - self.frame[1]) * (loot_coords[1] + i * loot_delta),
							 duration=0.5)
			pyautogui.moveRel(4, 0, duration=0.05)
			pyautogui.moveRel(-4, 0, duration=0.05)
			pyautogui.click()

	def watch_bait(self, bait_coords):
		# capturing of the float window
		bait_window = {'top': int(bait_coords[1] - self.bait_window / 2),
					   'left': int(bait_coords[0] - self.bait_window / 2),
					   'width': self.bait_window,
					   'height': self.bait_window}

		slope_samples_number = int(self.UI.slope_samples_edit.edit.text())

		score_buffer = deque(maxlen=slope_samples_number)
		slope_buffer = []
		std_scale = float(self.UI.bait_mov_sensibility_edit.edit.text())

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
					time.sleep(float(self.UI.post_detection_sleep_edit.edit.text()))
					pyautogui.rightClick()
					self.UI.log_viewer.emitter.emit("tried to capture fish")
					time.sleep(0.2)
					self.UI.log_viewer.emitter.emit("looting the fish...")
					self.loot()
					break

				self.display_bait_mask(current_bait)
				self.display_trigger_signal([slope, slope_threshold])

	def fish_grid(self):
		if self.dead_UI:
			exit()
		self.UI.log_viewer.emitter.emit("Throwing bait...")
		self.throw_bait()

		grid_step = 20
		found = False
		bait_coords = None

		a = int(self.frame[1] + self.frame[3] * self.grid_frac_ver[0])
		b = int(self.frame[1] + self.frame[3] * self.grid_frac_ver[1])
		c = int(self.frame[0] + self.frame[2] * self.grid_frac_hor[0])
		d = int(self.frame[0] + self.frame[2] * self.grid_frac_hor[1])

		for j in range(a, b, grid_step):
			if found:
				break
			for i in range(c, d, grid_step):
				precursor = win32gui.GetCursorInfo()[1]
				utils.move_mouse([i, j])
				time.sleep(0.02)
				postcursor = win32gui.GetCursorInfo()[1]
				if precursor != postcursor:
					found = True
					j += int(self.frame[2] / 100)
					pyautogui.moveRel(0, int(self.frame[2] / 100), duration=0.05)

					self.UI.log_viewer.emitter.emit("Found bait at coordinates {0} , {1}".format(i, j))

					bait_coords = [i, j]
					break
		if bait_coords is not None:
			self.watch_bait(bait_coords)

		self.tries += 1
		self.UI.tries_count_label.text_emitter.emit("{} tries".format(str(self.tries)))
		self.jump()

	def set_wow_frame(self, frame):
		self.frame = frame
		# dimension of the window that will encapsulate the bait
		self.bait_window = int(frame[3] / 5)

	def get_bait_mask(self, w):
		return self.process_bait(np.array(self.sct.grab(w)))

	@staticmethod
	def process_bait(img):
		return utils.binarize_canny(img)

	def display_bait_mask(self, bait_mask):
		self.UI.binary_image_widget.emitter.emit(np.rot90(bait_mask * 255, k=3))

	def display_trigger_signal(self, signals):
		self.UI.slope_signal_viewer.emitter.emit(signals)


if __name__ == "__main__":
	botUI = WowFishingBotUI()