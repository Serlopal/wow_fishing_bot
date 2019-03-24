import cv2
import pyautogui
import numpy as np
import win32gui
import sys
from mss import mss
import time
import utils
from threading import Thread
import os
from PyQt5.QtWidgets import QApplication, QGridLayout, QTextEdit, QMainWindow, QGroupBox, QAction, QDialog, QListWidget, \
QSizePolicy, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
pyautogui.PAUSE = 0.01
pyautogui.FAILSAFE = False


class QSignalViewer(pg.PlotWidget):
	emitter = pyqtSignal(object)

	def __init__(self, num_signals):
		super().__init__()
		# save number of signals
		self.nplots = num_signals
		# set number of samples to be displayed per signal at a time
		self.nsamples = 500
		# connect the signal to be emitted by the feeder to the slot of the plotWidget that will update the signals
		self.emitter.connect(lambda values: self.update(values))
		# buffer to store the data from all signals
		self.buff = np.zeros((self.nplots, self.nsamples))
		# create curves for the signals
		self.curves = []
		for i in range(self.nplots):
			c = pg.PlotCurveItem(pen=(i, self.nplots * 1.3))
			self.addItem(c)
			self.curves.append(c)

		#self.autoRange(padding = 0.05)

	def update(self, data):
		# update buffer
		self.buff = np.concatenate([self.buff[:, 1:], np.reshape(data, (-1, 1))], axis=1)
		# update plots
		for i in range(self.nplots):
			self.curves[i].setData(self.buff[i])

	def update_signals(self, values):
		self.emitter.emit(values)


class QLogger(QTextEdit):
	emitter = pyqtSignal(object)
	def __init__(self):
		super().__init__()
		self.emitter.connect(lambda text: self.update_log(text))

	def update_log(self, text):
		self.append("\n" + text)


class WowFishingBotUI():

	def __init__(self):
		self.program_name = 'WoW.exe'  # name of the process that runs WoW
		self.window_name = 'World of Warcraft'  # name of the WoW window in the taskbar
		self.startStop_fishing_flag = False
		self.create_UI()

		self.bot = WowFishingBot(self)
		self.create_menu()

		self.start_bot()

		self.start_UI()


	def create_menu(self):
		menubar = self.window.menuBar()
		fileMenu = menubar.addMenu('&File')
		exitAct = QAction("change parameters", self.window)
		exitAct.triggered.connect(self.show_reconfiguration_dialog)
		fileMenu.addAction(exitAct)

		self.startStop_fishing_action = QAction("Start Fishing", self.window)
		self.startStop_fishing_action.triggered.connect(self.startStop_fishing)
		fileMenu.addAction(self.startStop_fishing_action)


	def startStop_fishing(self):
		self.startStop_fishing_flag = False if self.startStop_fishing_flag else True
		self.startStop_fishing_action.setText("Stop fishing" if self.startStop_fishing_flag else "Start fishing")


	def show_reconfiguration_dialog(self):
		self.reconfiguration_dialog = QDialog()
		layout = QGridLayout()
		self.parameters = QListWidget()
		self.parameters.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.parameters.itemClicked.connect(self.display_parameter_info)

		self.parameters_dict = {

			"program_name":
				"Name of the process that is running wow. Needed to check the game is running.",

			"window_name":
				"Name of the Windows program window the game is running in. Needed to capture the screen it covers properly.",

			"gridFraction_horizontal":
				"Start and stop fraction of the horizontal length of the game's window in which the bait will be searched for",

			"gridFraction_vertical":
				"Start and stop fraction of the vertical length of the game's window in which the bait will be searched for",

			"fishing_hotkey":
				"Binding to the key action inside Wow",

			"loot_coords":
				"relative coordinates of the location in the screen of the loot window",

			"loot_coords_delta":
				"relative vertical distance between different possible objects inside the loot window",

			"bait_window":
				"Relative dimension of the square window where the bait is located, that will be watched to detect the fish at the right time",
		}
		for p in self.parameters_dict.keys():
			self.parameters.addItem(p)
		layout.addWidget(self.parameters, 0,0,10,5)

		# create textbox with description of parameter
		self.parametersInfo_label = QLabel(""" Click a command to show its info """)
		self.parametersInfo_label.setWordWrap(True)
		layout.addWidget(self.parametersInfo_label, 0,5,8,5)

		# create text box for the value
		self.parametersValue_label = QLineEdit(" Click a parameter to show its current value ")
		layout.addWidget(self.parametersValue_label, 8,5,2,3)

		# create save button
		save_parameter_button = QPushButton("Save")
		save_parameter_button.clicked.connect(self.save_parameter)
		layout.addWidget(save_parameter_button, 8, 8, 2, 3 )

		self.reconfiguration_dialog.setLayout(layout)
		self.reconfiguration_dialog.show()


	def save_parameter(self):
		param = self.parameters.currentItem().text()
		if hasattr(self, param):
			setattr(self, param, self.parametersValue_label.text())
		else:
			if isinstance(getattr(self.bot, param), str):
				setattr(self, param, self.parametersValue_label.text())
			elif isinstance(getattr(self.bot, param), list):
				setattr(self, param, map(float, self.parametersValue_label.text().split(" ")))
			elif isinstance(getattr(self.bot, param), float):
				setattr(self, param, float(self.parametersValue_label.text()))
			else:
				raise Exception("parameter tye unknown")


	def display_parameter_info(self, parameter):
		if hasattr(self, parameter.text()):
			param_value = getattr(self, parameter.text())
		else:
			param_value = getattr(self.bot, parameter.text())
			if isinstance(param_value, list):
				param_value = " ".join(map(str, param_value))

		param_info = self.parameters_dict[parameter.text()]
		self.parametersInfo_label.setText(param_info)
		self.parametersValue_label.setText(str(param_value))


	def create_UI(self):
		self.app = QApplication(["WowFishingBotUI"])
		self.window = QMainWindow()
		self.window.setGeometry(1100, 50, 800, 900 )
		self.signal_viewer = QSignalViewer(num_signals=1)
		self.log_viewer = QLogger()

		centralWidget = QGroupBox()
		layout = QGridLayout()
		layout.addWidget(self.signal_viewer, 0, 0, 10, 10)
		layout.addWidget(self.log_viewer, 10, 0, 10, 10)
		centralWidget.setLayout(layout)
		self.window.setCentralWidget(centralWidget)

		# register flag callback to let the backend know the UI has died
		self.app.aboutToQuit.connect(self.notify_dead_UI)


	def notify_dead_UI(self):
		self.bot.dead_UI = True


	def start_UI(self):
		self.window.show()
		self.app.exec_()


	def _start_bot(self):
		self.bot.find_wow()
		first = True
		while True:
			if self.startStop_fishing_flag:
				if first:
					self.log_viewer.emitter.emit("giving the user time to switch to WoW...")
					for i in reversed(range(10)):
						time.sleep(1)
						self.log_viewer.emitter.emit("{}".format(i))
					first = False
				self.bot.fish_grid()
			else:
				first = True


	def start_bot(self):
		thread = Thread(target=self._start_bot)
		thread.start()


class WowFishingBot():

	def __init__(self, UI):
		self.sct = mss()
		self.UI = UI
		self.dead_UI = False
		self.gridFraction_horizontal = [0.3, 0.7]
		self.gridFraction_vertical = [0.1, 0.4]
		self.fishing_hotkey = '8'  # hotkey assigned in the game to the action of fishing

		self.loot_coords = [0.053, 0.28]  # relative coordinates of the location in the screen of the loot window
		self.loot_coords_delta = 0.03  # vertical movement between different objects in the loot window

		self.tries = 0

	def make_screenshot(self):
		color_frame = self.color_frame = self.sct.grab(self.frame)
		return cv2.cvtColor(np.array(color_frame), cv2.COLOR_RGB2GRAY)


	def throw_bait(self):
		pyautogui.hotkey(self.fishing_hotkey)


	def jump(self):
		self.UI.log_viewer.emitter.emit('Jump!')
		pyautogui.hotkey(' ')
		time.sleep(1)


	def loot(self):
		nitems = 3
		for i in range(nitems):
			time.sleep(0.5)
			pyautogui.moveTo(x=self.frame[0] + (self.frame[2] - self.frame[0]) * self.loot_coords[0],
							 y=self.frame[1] + (self.frame[3] - self.frame[1]) * (self.loot_coords[1] + i * self.loot_coords_delta),
							 duration=0.5)
			pyautogui.moveRel(4, 0, duration=0.05)
			pyautogui.moveRel(-4, 0, duration=0.05)
			pyautogui.click()


	def find_wow(self):
		# check Wow is running
		if utils.check_process(self.UI.program_name):
			self.frame = utils.get_window(self.UI.window_name)
			self.UI.log_viewer.emitter.emit("Wow window at " + str(self.frame))
			self.bait_window = int(self.frame[3]/20)  # dimension of the window that will encapsulate the bait
		else:
			self.UI.log_viewer.emitter.emit("Wow not found running")
			sys.exit()


	def watch_bait(self, bait_coords):
		# capturing of the float window
		bait_window = {'top': int(bait_coords[1] - self.bait_window / 2), 'left': int(bait_coords[0] - self.bait_window / 2),
				  'width': self.bait_window, 'height': self.bait_window}
		bait_prior   = utils.binarize(utils.apply_kmeans_colors(np.array(self.sct.grab(bait_window))))

		# list with all the differences between sampled images
		all_diffs = []
		avg_diff = 0
		std_diff = 0
		done_getting_stats = False
		self.UI.log_viewer.emitter.emit("watching float...")
		t = time.time()
		while time.time() - t < 30: # fishing process takes 30 secs
			float_current = utils.binarize(utils.apply_kmeans_colors(np.array(self.sct.grab(bait_window))))
			diff = np.sum(np.multiply(float_current, bait_prior))
			# emit difference to signal viewer
			self.UI.signal_viewer.emitter.emit(diff)
			if time.time() - t > 4 and not done_getting_stats:
				avg_diff = np.mean(all_diffs)
				std_diff = np.std(all_diffs)
				done_getting_stats = True

			if time.time() - t > 4:
				if diff < avg_diff - 3 * std_diff:
					pyautogui.rightClick()
					self.UI.log_viewer.emitter.emit("tried to capture fish")
					time.sleep(0.2)
					self.UI.log_viewer.emitter.emit("looting the fish...")
					self.loot()
					break
			bait_prior = float_current
			all_diffs.append(diff)


	def fish_grid(self):
		if self.dead_UI:
			exit()
		self.UI.log_viewer.emitter.emit("Throwing bait...")
		self.throw_bait()
		time.sleep(3)

		grid_step = 20
		found = False
		bait_coords = None
		for j in range(int(self.frame[1] + self.frame[3] * self.gridFraction_vertical[0]), int(self.frame[1] + self.frame[3] * self.gridFraction_vertical[1]), grid_step):
			if found:
				break
			for i in range(int(self.frame[0] + self.frame[2] * self.gridFraction_horizontal[0]) , int(self.frame[0] + self.frame[2] * self.gridFraction_horizontal[1]), grid_step):
				precursor = win32gui.GetCursorInfo()[1]
				utils.move_mouse([i, j])
				time.sleep(0.02)
				postcursor = win32gui.GetCursorInfo()[1]
				if precursor != postcursor:
					found = True
					self.UI.log_viewer.emitter.emit("Found bait at coordinates {0} , {1}".format(i,j))
					bait_coords = [i, j]
					break
		if bait_coords is not None:
			self.watch_bait(bait_coords)
			self.tries += 1
			self.UI.log_viewer.emitter.emit("Fishing try number {}".format(str(self.tries)))

		self.jump()


if __name__ == "__main__":
	botUI = WowFishingBotUI()
