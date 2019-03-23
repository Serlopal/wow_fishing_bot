import cv2
import pyscreenshot as ImageGrab
import pyautogui
import numpy as np
import psutil
import win32gui
import sys
from mss import mss
import mss.tools as msstools

from sklearn.cluster import DBSCAN
import win32api, win32con
import time
import utils
# from tensorforce.agents import DQNAgent
from matplotlib import pyplot as plt
from threading import Thread
import os
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QMainWindow,\
	QGraphicsView, QGraphicsScene, QVBoxLayout, QHBoxLayout, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import PyQt5.QtCore as QtCore
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

	def update(self, data):
		# update buffer
		self.buff = np.concatenate([self.buff[:, 1:], np.reshape(data, (-1, 1))], axis=1)
		# update plots
		for i in range(self.nplots):
			self.curves[i].setData(self.buff[i])

	def update_signals(self, values):
		self.emitter.emit(values)


class QImshow(QGraphicsView):
	emitter = pyqtSignal(object)

	def __init__(self):
		super().__init__()
		self._figure = plt.figure()
		self.emitter.connect(self.update_figure)

		self.scene = QGraphicsScene(self)
		self.setScene(self.scene)
		self.canvas = FigureCanvas(self._figure)
		self.scene.addWidget(self.canvas)
		self.canvas.show()

	def update(self):
		self.fitInView(self.scene.sceneRect())
		self.canvas.draw()
		self.canvas.show()

	def update_figure(self, figure):
		# self.fitInView(self.scene.sceneRect())
		self.canvas.figure = figure
		self.update ()

	@property
	def figure(self):
		return self.figure

	@figure.setter
	def figure(self, figure):
		self.emitter.emit(figure)


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
		self.fishing_hotkey = '8'  # hotkey assigned in the game to the action of fishing
		self.focusw = [50, 100, 200, 200]  # 0:-2, 1:-3 number of pixels removed from the outer of the image where we know the bait wont be
		self.loot_coords = [0.053, 0.28]  # relative coordinates of the location in the screen of the loot window
		self.loot_coords_delta = 0.03  # vertical movement between different objects in the loot window

		self.loot_window_coords = [0.17, 0.01, 0.33, 0.181]  # 0:0+2, 1:1+3
		self.loot_window = None
		self.tries = 0
		self.bait_window = 50  # dimension of the window that will encapsule the float

		self.create_UI()

		self.bot = WowFishingBot(self)
		self.bot.find_wow()
		self.start_bot()

		self.start_UI()

	def create_UI(self):
		self.app = QApplication(["WowFishingBotUI"])
		self.window = QWidget()
		self.window.setGeometry(1100, 50, 800, 900 )
		self.bot_plotter = QImshow()
		self.signal_viewer = QSignalViewer(num_signals=1)
		self.log_viewer = QLogger()
		layout = QGridLayout()
		layout.addWidget(self.bot_plotter, 0, 0, 10, 10)
		layout.addWidget(self.signal_viewer, 0, 10, 10, 10)
		layout.addWidget(self.log_viewer, 10, 0, 10, 20)

		self.window.setLayout(layout)

		# register flag callback to let the backend know the UI has died
		self.app.aboutToQuit.connect(self.notify_dead_UI)

	def notify_dead_UI(self):
		self.bot.dead_UI = True

	def start_UI(self):
		self.window.show()
		self.app.exec_()

	def _start_bot(self):
		self.bot.find_wow()
		self.log_viewer.emitter.emit("giving the user time to set UI")
		for i in reversed(range(15)):
			time.sleep(1)
			self.log_viewer.emitter.emit("{}".format(i))

		while True:
			self.bot.fish()

	def start_bot(self):
		thread = Thread(target=self._start_bot)
		thread.start()


class WowFishingBot():

	def __init__(self, UI):
		self.sct = mss()
		self.UI = UI
		self.dead_UI = False

	def make_screenshot(self):
		color_frame = self.color_frame = self.sct.grab(self.window)
		return cv2.cvtColor(np.array(color_frame), cv2.COLOR_RGB2GRAY)


	def throw_bait(self, fishing_hotkey):
		pyautogui.hotkey(fishing_hotkey)


	def jump(self):
		self.UI.log_viewer.emitter.emit('Jump!')
		pyautogui.hotkey(' ')
		time.sleep(1)


	def look4object(self, frame, object):
		self.UI.log_viewer.emitter.emit ('Looking for bait...')
		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()
		mask = np.zeros_like(frame)
		mask[0: int(frame.shape[0] * 0.8), int(frame.shape[1] * 0.33): int(frame.shape[1] * 0.66)] = 255
		# find the keypoints and descriptors with SIFT for both current frame and bait template
		kp_frame, des_frame = sift.detectAndCompute(frame, mask=mask)
		kp_template, des_template= sift.detectAndCompute(object, mask=None)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		# get matches of the bait in the current frame
		matches = flann.knnMatch(des_frame, des_template, k=2)

		if not matches: return None, None

		# extract matching points
		src_pts = np.float32([kp_frame[m.queryIdx].pt for m, _ in matches]).reshape(-1, 2)
		# find float cluster of matching points
		bait_location = utils.find_highest_density(src_pts, 15)

		# build image with bait candidates by SIFT and use a signal to emit it to the UI
		SIFT_frame = plt.figure()
		plt.imshow(frame, figure = SIFT_frame)
		for pt in src_pts:
			plt.scatter(pt[0], pt[1], s=10, c="red", figure = SIFT_frame)
		plt.scatter(bait_location[0], bait_location[1], s=10, c="blue", figure = SIFT_frame)
		plt.axis('off')
		self.UI.bot_plotter.figure = SIFT_frame

		return bait_location +  np.array(self.window[0:2]) #+ np.array(self.UI.focusw)[1::-1]


	def loot(self):
		nitems = 3
		for i in range(nitems):
			time.sleep(1)
			pyautogui.moveTo(x=self.window[0] + (self.window[2] - self.window[0]) * self.UI.loot_coords[0],
							 y=self.window[1] + (self.window[3] - self.window[1]) * (self.UI.loot_coords[1] + i * self.UI.loot_coords_delta),
							 duration=1)
			pyautogui.moveRel(4, 0, duration=0.2)
			pyautogui.moveRel(-4, 0, duration=0.2)
			pyautogui.click()


	def find_wow(self):
		# check Wow is running
		if utils.check_process(self.UI.program_name):
			self.window = utils.get_window(self.UI.window_name)
			self.UI.log_viewer.emitter.emit("Wow window at " + str(self.window))
			self.UI.log_viewer.emitter.emit("Waiting 2 seconds, so you can switch to WoW")
			time.sleep(2)
			self.jump()
		else:
			self.UI.log_viewer.emitter.emit("Wow not found running, exiting...")
			sys.exit()


	def watch_bait(self, bait_coords):
		# capturing of the float window
		bait_window = {'top': int(bait_coords[1] - self.UI.bait_window / 2), 'left': int(bait_coords[0] - self.UI.bait_window / 2),
				  'width': self.UI.bait_window, 'height': self.UI.bait_window}
		bait_prior   = utils.binarize(utils.apply_kmeans_colors(np.array(self.sct.grab(bait_window))))

		# list with all the differences between sampled images
		all_diffs = []
		avg_diff = 0
		std_diff = 0
		c = 0
		self.UI.log_viewer.emitter.emit("watching float...")
		t = time.time()
		while time.time() - t < 30: # fishing process takes 30 secs
			float_current = utils.binarize(utils.apply_kmeans_colors(np.array(self.sct.grab(bait_window))))
			diff = np.sum(np.multiply(float_current, bait_prior))
			# emit difference to signal viewer
			self.UI.signal_viewer.emitter.emit(diff)
			if c == 200:
				avg_diff = np.mean(all_diffs)
				std_diff = np.std(all_diffs)

			if c > 200:
				if diff < avg_diff - 3 * std_diff:
					pyautogui.rightClick()
					self.UI.log_viewer.emitter.emit("tried to capture fish")
					time.sleep(0.2)
					self.UI.log_viewer.emitter.emit("looting the fish...")
					self.loot()
					break
			bait_prior = float_current
			all_diffs.append(diff)
			c += 1


	def fish(self):
		if self.dead_UI:
			exit()
		self.UI.log_viewer.emitter.emit("Throwing bait...")
		self.throw_bait(self.UI.fishing_hotkey)
		# find fishing float in image
		self.UI.log_viewer.emitter.emit("waiting for bait to be floating...")
		time.sleep(2)
		self.UI.log_viewer.emitter.emit("Trying to find bait...")
		# frame = utils.get_subwindow(self.make_screenshot(self.window), self.UI.focusw, "pos2neg")
		frame = self.make_screenshot()
		bait_coords = self.look4object(frame=frame, object = cv2.imread('var/fishing_float_9.png', 0))

		# if we cannot find the float, try again
		if bait_coords is None or np.any(np.isnan(bait_coords)):
			self.UI.log_viewer.emitter.emit("Could not find bait :(")
			self.jump()
			return


		self.UI.log_viewer.emitter.emit("Found bait at {}, moving mouse to it".format(bait_coords))
		utils.move_mouse(bait_coords.tolist())


		time.sleep(2)

		self.watch_bait(bait_coords)
		self.UI.tries += 1
		self.UI.log_viewer.emitter.emit("Fishing try number {}".format(str(self.UI.tries)))

		self.jump()


if __name__ == "__main__":
	botUI = WowFishingBotUI()
