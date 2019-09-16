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
QSizePolicy, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
pyautogui.PAUSE = 0.01
pyautogui.FAILSAFE = False


class Log(QTextEdit):
	emitter = pyqtSignal(object)

	def __init__(self):
		super().__init__()
		self.emitter.connect(lambda text: self.update_log(text))

	def update_log(self, text):
		self.append("\n" + text)


class LabeledLineEdit(QGroupBox):

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

		self.create_ui()

		self.bot = WowFishingBot(self)

		self.start_ui()

	def create_ui(self):
		# create QT application and main window
		self.app = QApplication(["WowFishingBotUI"])
		self.window = QMainWindow()
		self.window.setGeometry(1100, 50, 800, 900)

		# create central widget and layout
		central_widget = QGroupBox()
		layout = QGridLayout()

		# CONTROLS AND TOGGLES
		# edit to change games process name
		self.game_process_name = LabeledLineEdit("WoW's process name: ", "Wow.exe")
		layout.addWidget(self.game_process_name, 0, 0, 1, 1)
		# edit to change games window name
		self.window_name = LabeledLineEdit("WoW's window name: ", "World of Warcraft")
		layout.addWidget(self.window_name, 0, 1, 1, 1)

		# button to try to find the process from the line edit
		self.find_wow_button = QPushButton("Find WOW!")
		self.find_wow_button.clicked.connect(self.find_wow)
		layout.addWidget(self.find_wow_button, 0, 2, 1, 1)

		# toggle for loop fishing
		self.auto_fish_toggle = QCheckBox("Auto pilot")
		layout.addWidget(self.auto_fish_toggle, 1, 0, 1, 1)

		# Fish! button
		self.fish_button = QPushButton("Fish")
		self.fish_button.clicked.connect(self._fish)
		self.fish_button.setEnabled(False)
		layout.addWidget(self.fish_button, 1, 1, 1, 2)

		# LOG FROM BOT ACTIVITY
		self.log_viewer = Log()
		layout.addWidget(self.log_viewer, 10, 0, 10, 10)
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
			if not self.auto_fish_toggle:
				break

	def start_bot(self):
		thread = Thread(target=self._fish)
		thread.start()

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

	def __init__(self, UI):
		self.sct = mss()
		self.UI = UI
		self.dead_UI = False
		self.gridFraction_horizontal = [0.3, 0.7]
		self.gridFraction_vertical = [0.1, 0.4]
		self.fishing_hotkey = '8'  # hotkey assigned in the game to the action of fishing
		self.frame = None
		self.bait_window = None

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
		if utils.check_process(self.UI.game_process_name.text()):
			self.frame = utils.get_window(self.UI.window_name)
			self.UI.log_viewer.emitter.emit("Wow window at " + str(self.frame))
			self.bait_window = int(self.frame[3]/20)  # dimension of the window that will encapsulate the bait
		else:
			self.UI.log_viewer.emitter.emit("Wow not found running")

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
					j += int(self.frame[2] / 100)
					pyautogui.moveRel(0, int(self.frame[2] / 100), duration=0.05)

					self.UI.log_viewer.emitter.emit("Found bait at coordinates {0} , {1}".format(i,j))

					bait_coords = [i, j]
					break
		if bait_coords is not None:
			# lower a bit the window of the bait. The cursor grid from top to bottom so usually it will not be centered vertically with the bait
			self.watch_bait(bait_coords)
			self.tries += 1
			self.UI.log_viewer.emitter.emit("Fishing try number {}".format(str(self.tries)))

		self.jump()

	def set_wow_frame(self, frame):
		self.frame = frame
		# dimension of the window that will encapsulate the bait
		self.bait_window = int(frame[3] / 20)


if __name__ == "__main__":
	botUI = WowFishingBotUI()
