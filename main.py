import cv2
import pyscreenshot as ImageGrab
import pyautogui
import numpy as np
import psutil
import win32gui
import sys
from mss import mss
from sklearn.cluster import DBSCAN
import win32api, win32con
import time
import utils
from tensorforce.agents import DQNAgent
from matplotlib import pyplot as plt
from threading import Thread
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


pyautogui.PAUSE = 0.01
pyautogui.FAILSAFE = False

class WowFishingBot():
	def __init__(self):
		print("setting up bot...")
		self.program_name = 'WoW.exe' # name of the process that runs WoW
		self.window_name = 'World of Warcraft' # name of the WoW window in the taskbar
		self.fishing_hotkey = '8' # hotkey assigned in the game to the action of fishing
		self.focusw = [50, 100, 200, 200] # 0:-2, 1:-3 number of pixels removed from the outer of the image where we know the bait wont be
		self.sct = mss() # mss class instance to capture the screen at high frequencies
		self.loot_coords = [0.053, 0.28] # relative coordinates of the location in the screen of the loot window
		self.loot_coords_delta = 0.03 # vertical movement between different objects in the loot window

		self.loot_window_coords = [0.17, 0.01, 0.33, 0.181] # 0:0+2, 1:1+3
		self.loot_window = None
		self.tries = 0
		self.bait_window = 50 # dimension of the window that will encapsule the float
		### RL
		# Network is an ordered list of layers
		network_spec = [
			{
				"type": "conv2d",
				"size": 32,
				"window": 8,
				"stride": 4
			},
			{
				"type": "conv2d",
				"size": 64,
				"window": 4,
				"stride": 2
			},
			{
				"type": "flatten"
			},
			{
				"activation": "sigmoid",
				"type": "dense",
				"size": 2
			}
		]
		# Define a state
		states = dict(shape=(64, 64, 1), type='float')
		# Define an action
		actions = dict(shape=(1, 2), type='int', num_actions=1)

		update_mode = dict(
			unit='timesteps',
			batch_size=1,
			frequency=4
		)


		preprocessing = [
			{
				"type": "image_resize",
				"width": 64,
				"height": 64
			},
			{
				"type": "normalize"
			}
		]
		self.agent = DQNAgent(
			states=states,
			actions=actions,
			network=network_spec,
			states_preprocessing = preprocessing,
			update_mode=update_mode
		)

	# GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

	def make_screenshot(self, window):
		return cv2.cvtColor(np.array(self.sct.grab(self.window)), cv2.COLOR_RGB2GRAY)


	def throw_bait(self, fishing_hotkey):
		pyautogui.hotkey(fishing_hotkey)


	def jump(self):
		print('Jump!')
		pyautogui.hotkey(' ')
		time.sleep(1)


	def look4object(self, frame, object):
		print ('Looking for bait...')
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
		# store all the good matches as per Lowe's ratio test.
		good = list(zip(*matches))[0] # same as good = [m for m, _ in matches]
		# compute matching points
		src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 2)
		# find float cluster of matching points
		db = DBSCAN(eps=20, min_samples=3).fit(src_pts)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_
		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		unique_labels = element = next(iter(set(labels)))
		class_member_mask = (labels == unique_labels)
		bait_location = src_pts[class_member_mask & core_samples_mask]


		return np.mean(bait_location, axis=0) + np.array(self.window[0:2]) + np.array(self.focusw)[1::-1], src_pts


	def loot(self):
		nitems = 3
		for i in range(nitems):
			time.sleep(1)
			pyautogui.moveTo(x=self.window[0] + (self.window[2] - self.window[0]) * self.loot_coords[0],
							 y=self.window[1] + (self.window[3] - self.window[1]) * (self.loot_coords[1] + i * self.loot_coords_delta),
							 duration=1)
			pyautogui.moveRel(4, 0, duration=0.2)
			pyautogui.moveRel(-4, 0, duration=0.2)
			pyautogui.click()


	def find_wow(self):
		# check Wow is running
		if utils.check_process(self.program_name):
			self.window = utils.get_window(self.window_name)
			print("Wow window at " + str(self.window))
			print("Waiting 2 seconds, so you can switch to WoW")
			time.sleep(2)
			self.jump()
		else:
			print("Wow not found running, exiting...")
			sys.exit()


	def check_captured_something(self):
		img = self.make_screenshot(self.window)
		loot_window = utils.get_subwindow(img, self.loot_window_coords, "from_corner")
		plt.imshow(loot_window)
		plt.show()


	def _watch_loot_window(self):
		# reset trigger flag
		self.found_lootw = False
		# get samples
		samples = []
		sct = mss()

		for _ in range(30):
			img = cv2.cvtColor(np.array(sct.grab(self.window)), cv2.COLOR_RGB2GRAY)
			samples.append(utils.get_subwindow(img, self.loot_window_coords, "from_corner"))
			time.sleep(0.1)
		avg_diff = np.mean([np.sum(np.abs(np.subtract(samples[i], samples[i+1]))) for i in range(len(samples)-1)])
		std_diff = np.std([np.sum(np.abs(np.subtract(samples[i], samples[i+1]))) for i in range(len(samples)-1)])
		# now watch for loot window popping up
		prior_loot = samples[-1]
		while True:
			img = img = cv2.cvtColor(np.array(sct.grab(self.window)), cv2.COLOR_RGB2GRAY)
			curr_loot = utils.get_subwindow(img, self.loot_window_coords, "from_corner")
			diff = np.sum(np.abs(np.subtract(curr_loot, prior_loot)))
			if diff > avg_diff + 3 * std_diff:
				self.found_lootw = True
				break
			prior_loot = curr_loot


	def watch_loot_window(self):
		thread = Thread(target=self._watch_loot_window)
		thread.start()


	def watch_bait(self, bait_coords):
		# capturing of the float window
		bait_window = {'top': int(bait_coords[1] - self.bait_window / 2), 'left': int(bait_coords[0] - self.bait_window / 2),
				  'width': self.bait_window, 'height': self.bait_window}
		bait_prior   = utils.binarize(utils.aply_kmeans_colors(np.array(self.sct.grab(bait_window))))

		# list with all the differences between sampled images
		all_diffs = []
		avg_diff = 0
		std_diff = 0
		c = 0
		print("watching float...")
		while c < 1400:
			float_current = utils.binarize(utils.aply_kmeans_colors(np.array(self.sct.grab(bait_window))))
			diff = np.sum(np.multiply(float_current, bait_prior))

			if c == 200:
				avg_diff = np.mean(all_diffs)
				std_diff = np.std(all_diffs)

			if c > 200:
				if diff < avg_diff - 3 * std_diff:
					pyautogui.rightClick()
					print("tried to capture fish")
					time.sleep(0.2)
					# if self.found_lootw:
					# 	print("SUCCEDED CAPTURING THE FISH")
					# loot
					print("looting the fish...")
					self.loot()
					break
			bait_prior = float_current
			all_diffs.append(diff)
			c += 1


	def fish_manual(self):
		print("Throwing bait...")
		self.throw_bait(self.fishing_hotkey)
		frame = utils.get_subwindow(self.make_screenshot(self.window), self.focusw, "pos2neg")

		# find fishing float in image
		print("Trying to find bait...")
		bait_coords, _ = self.look4object(frame=frame, object = cv2.imread('var/fishing_float_9.png', 0))

		# if we cannot find the float, try again
		if bait_coords is None or np.any(np.isnan(bait_coords)):
			print("Could not find bait :(")
			self.jump()
			return

		# get normal cursor info
		normal_cursor = win32gui.GetCursorInfo()
		print("normal cursor {}".format(normal_cursor))
		time.sleep(0.5)

		print("Found bait at {}, moving mouse to it".format(bait_coords))
		utils.move_mouse(bait_coords.tolist())

		# check if the bait is really there by checking if the
		time.sleep(0.5)
		gear_cursor = win32gui.GetCursorInfo()
		print("GEAR cursor {}".format(gear_cursor))


		self.watch_loot_window()

		self.watch_bait(bait_coords)

		self.tries += 1
		print("Fishing try number {}".format(str(self.tries)))

		self.jump()


	def fish_RL(self):

		print("Booting up...")

		self.throw_bait(self.fishing_hotkey)
		frame = utils.get_subwindow(self.make_screenshot(self.window), self.focusw, "pos2neg")
		#plt.imshow(frame)
		#plt.show()
		# find fishing float in image
		print("Throwing bait...")
		action = np.squeeze(self.agent.act(frame))
		wind0w_main_coords = [frame.shape[0] + self.focusw[0], frame.shape[1] + self.focusw[1]]
		bait_coords = np.multiply(action, wind0w_main_coords)

		# get normal cursor info
		normal_cursor = win32gui.GetCursorInfo()[1]
		# print("normal cursor {}".format(normal_cursor))
		time.sleep(0.5)

		print("Found bait at {}, moving mouse to it".format(bait_coords))
		utils.move_mouse(bait_coords.tolist())

		# check if the bait is really there by checking if the
		time.sleep(0.5)
		gear_cursor = win32gui.GetCursorInfo()[1]
		# print("GEAR cursor {}".format(gear_cursor))

		reward = 1 if normal_cursor != gear_cursor else 0
		self.agent.observe(reward=reward, terminal=False)


		# self.watch_bait(bait_coords)



class WowFishingBotRL():
	def __init__(self):
		print("setting up bot...")
		self.program_name = 'WoW.exe' # name of the process that runs WoW
		self.window_name = 'World of Warcraft' # name of the WoW window in the taskbar
		self.fishing_hotkey = '8' # hotkey assigned in the game to the action of fishing
		self.focusw = [50, 100, 200, 200] # 0:-2, 1:-3 number of pixels removed from the outer of the image where we know the bait wont be
		self.sct = mss() # mss class instance to capture the screen at high frequencies
		self.loot_coords = [0.053, 0.28] # relative coordinates of the location in the screen of the loot window
		self.loot_coords_delta = 0.03 # vertical movement between different objects in the loot window

		self.loot_window_coords = [0.17, 0.01, 0.33, 0.181] # 0:0+2, 1:1+3
		self.loot_window = None
		self.tries = 0
		self.bait_window = 50 # dimension of the window that will encapsule the float
		### RL
		# Network is an ordered list of layers
		network_spec = [
			{
				"type": "conv2d",
				"size": 32,
				"window": 8,
				"stride": 4
			},
			{
				"type": "conv2d",
				"size": 64,
				"window": 4,
				"stride": 2
			},
			{
				"type": "flatten"
			},
			{
				"activation": "sigmoid",
				"type": "dense",
				"size": 2
			}
		]
		# Define a state
		states = dict(shape=(64, 64, 1), type='float')
		# Define an action
		actions = dict(shape=(1, 2), type='int', num_actions=1)

		update_mode = dict(
			unit='timesteps',
			batch_size=1,
			frequency=4
		)


		preprocessing = [
			{
				"type": "image_resize",
				"width": 64,
				"height": 64
			},
			{
				"type": "normalize"
			}
		]
		self.agent = DQNAgent(
			states=states,
			actions=actions,
			network=network_spec,
			states_preprocessing = preprocessing,
			update_mode=update_mode
		)

	# GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

	def make_screenshot(self, window):
		return cv2.cvtColor(np.array(self.sct.grab(self.window)), cv2.COLOR_RGB2GRAY)


	def throw_bait(self, fishing_hotkey):
		pyautogui.hotkey(fishing_hotkey)


	def jump(self):
		print('Jump!')
		pyautogui.hotkey(' ')
		time.sleep(1)


	def loot(self):
		nitems = 3
		for i in range(nitems):
			time.sleep(1)
			pyautogui.moveTo(x=self.window[0] + (self.window[2] - self.window[0]) * self.loot_coords[0],
							 y=self.window[1] + (self.window[3] - self.window[1]) * (self.loot_coords[1] + i * self.loot_coords_delta),
							 duration=1)
			pyautogui.moveRel(4, 0, duration=0.2)
			pyautogui.moveRel(-4, 0, duration=0.2)
			pyautogui.click()


	def find_wow(self):
		# check Wow is running
		if utils.check_process(self.program_name):
			self.window = utils.get_window(self.window_name)
			print("Wow window at " + str(self.window))
			print("Waiting 2 seconds, so you can switch to WoW")
			time.sleep(2)
			self.jump()
		else:
			print("Wow not found running, exiting...")
			sys.exit()


	def check_captured_something(self):
		img = self.make_screenshot(self.window)
		loot_window = utils.get_subwindow(img, self.loot_window_coords, "from_corner")
		plt.imshow(loot_window)
		plt.show()


	def _watch_loot_window(self):
		# reset trigger flag
		self.found_lootw = False
		# get samples
		samples = []
		sct = mss()

		for _ in range(30):
			img = cv2.cvtColor(np.array(sct.grab(self.window)), cv2.COLOR_RGB2GRAY)
			samples.append(utils.get_subwindow(img, self.loot_window_coords, "from_corner"))
			time.sleep(0.1)
		avg_diff = np.mean([np.sum(np.abs(np.subtract(samples[i], samples[i+1]))) for i in range(len(samples)-1)])
		std_diff = np.std([np.sum(np.abs(np.subtract(samples[i], samples[i+1]))) for i in range(len(samples)-1)])
		# now watch for loot window popping up
		prior_loot = samples[-1]
		while True:
			img = img = cv2.cvtColor(np.array(sct.grab(self.window)), cv2.COLOR_RGB2GRAY)
			curr_loot = utils.get_subwindow(img, self.loot_window_coords, "from_corner")
			diff = np.sum(np.abs(np.subtract(curr_loot, prior_loot)))
			if diff > avg_diff + 3 * std_diff:
				self.found_lootw = True
				break
			prior_loot = curr_loot


	def watch_loot_window(self):
		thread = Thread(target=self._watch_loot_window)
		thread.start()


	def watch_bait(self, bait_coords):
		# capturing of the float window
		bait_window = {'top': int(bait_coords[1] - self.bait_window / 2), 'left': int(bait_coords[0] - self.bait_window / 2),
				  'width': self.bait_window, 'height': self.bait_window}
		bait_prior   = utils.binarize(utils.aply_kmeans_colors(np.array(self.sct.grab(bait_window))))

		# list with all the differences between sampled images
		all_diffs = []
		avg_diff = 0
		std_diff = 0
		c = 0
		print("watching float...")
		while c < 1400:
			float_current = utils.binarize(utils.aply_kmeans_colors(np.array(self.sct.grab(bait_window))))
			diff = np.sum(np.multiply(float_current, bait_prior))

			if c == 200:
				avg_diff = np.mean(all_diffs)
				std_diff = np.std(all_diffs)

			if c > 200:
				if diff < avg_diff - 3 * std_diff:
					pyautogui.rightClick()
					print("tried to capture fish")
					time.sleep(0.2)
					# if self.found_lootw:
					# 	print("SUCCEDED CAPTURING THE FISH")
					# loot
					print("looting the fish...")
					self.loot()
					break
			bait_prior = float_current
			all_diffs.append(diff)
			c += 1


	def fish(self):

		print("Booting up...")

		self.throw_bait(self.fishing_hotkey)
		frame = utils.get_subwindow(self.make_screenshot(self.window), self.focusw, "pos2neg")
		#plt.imshow(frame)
		#plt.show()
		# find fishing float in image
		print("Throwing bait...")
		action = np.squeeze(self.agent.act(frame))
		wind0w_main_coords = [frame.shape[0] + self.focusw[0], frame.shape[1] + self.focusw[1]]
		bait_coords = np.multiply(action, wind0w_main_coords)

		# get normal cursor info
		normal_cursor = win32gui.GetCursorInfo()[1]
		# print("normal cursor {}".format(normal_cursor))
		time.sleep(0.5)

		print("Found bait at {}, moving mouse to it".format(bait_coords))
		utils.move_mouse(bait_coords.tolist())

		# check if the bait is really there by checking if the
		time.sleep(0.5)
		gear_cursor = win32gui.GetCursorInfo()[1]
		# print("GEAR cursor {}".format(gear_cursor))

		reward = 1 if normal_cursor != gear_cursor else 0
		self.agent.observe(reward=reward, terminal=False)














if __name__ == "__main__":

	bot = WowFishingBot()
	bot.find_wow()
	while True:
		bot.fish_RL()