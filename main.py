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

pyautogui.PAUSE = 0.01


class WowFishingBot():
	def __init__(self):
		print("setting up bot...")
		self.program_name = 'WoW.exe' # name of the process that runs WoW
		self.window_name = 'World of Warcraft' # name of the WoW window in the taskbar
		self.fishing_hotkey = '8' # hotkey assigned in the game to the action of fishing
		self.focusw = 50 # size in pixels of the window used to monitor the fishing float
		self.sct = mss() # mss class instance to capture the screen at high frequencies
		self.loot_coords = [0.053, 0.28] # relative coordinates of the location in the screen of the loot window
		self.loot_coords_delta = 0.03 # vertical movement between different objects in the loot window


	def send_float(self, fishing_hotkey):
		print('Sending float')
		pyautogui.hotkey(fishing_hotkey)
		print('Float is sent, waiting animation')
		time.sleep(3)


	def jump(self):
		print('Jump!')
		pyautogui.hotkey(' ')
		time.sleep(1)


	def find_float(self, img):
		print ('Looking for float...')
		template = cv2.imread('var/fishing_float_9.png', 0)  # trainImage
		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()
		mask = np.zeros_like(img)
		mask[0: int(img.shape[0]*0.8), int(img.shape[1]*0.33) : int(img.shape[1]*0.66)] = 255
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img, mask=mask)
		kp2, des2 = sift.detectAndCompute(template, mask=None)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		search_params = dict(checks=50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k=2)
		# store all the good matches as per Lowe's ratio test.
		good = list(zip(*matches))[0] # same as good = [m for m, _ in matches]
		# compute matching points
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
		# find float cluster of matching points
		db = DBSCAN(eps=20, min_samples=3).fit(src_pts)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_
		# Number of clusters in labels, ignoring noise if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		unique_labels = element = next(iter(set(labels)))
		class_member_mask = (labels == unique_labels)
		xy = src_pts[class_member_mask & core_samples_mask]

		return np.mean(xy, axis=0) + np.array(self.window[0:2]) + self.focusw


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


	def fish(self):
		print("Booting up...")

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


		while True:
			self.send_float(self.fishing_hotkey)
			img = np.array(utils.make_screenshot(self.window).convert('L'))[self.focusw:-self.focusw, self.focusw:-self.focusw]


			# find fishing float in image
			print("Throwing float...")
			float_coords = self.find_float(img)

			# if we cannot find the float, try again
			print(float_coords)
			if float_coords is None or np.any(np.isnan(float_coords)):
				print("Could not find float, retrying...")
				self.jump()
				continue

			print("Found float, moving cursor to it...")
			utils.move_mouse(float_coords.tolist())
			old_diff = 99999
			# counter of sampled images of the float
			c = 0

			# dimension of the window that will encapsule the float
			window_dim = 50
			# capturing of the float window
			window_float = {'top': int(float_coords[1] - window_dim / 2), 'left': int(float_coords[0] - window_dim / 2),
					  'width': window_dim, 'height': window_dim}
			float_prior   = utils.binarize(utils.kmeans_centroids(np.array(self.sct.grab(window_float))))
			# float_prior = binarize(kmeans_apply(np.array(sct.grab(window_float)),centroids))



			# number of samples to consider to measure the change rate of the image across time
			window_length = 1
			# list to store the change rates
			last_diffs = list(np.zeros(window_length))
			# list with all the differences between sampled images
			all_diffs = []
			avg_diff = 0
			std_diff = 0

			print("watching float...")
			while c < 1400 :
				float_current = utils.binarize(utils.kmeans_centroids(np.array(self.sct.grab(window_float))))
				diff = np.sum(np.multiply(float_current, float_prior))

				if c == 200:
					avg_diff = np.mean(all_diffs)
					std_diff = np.std(all_diffs)

				if c > 200:
					# print("c: " + str(c) + " diff: " + str(diff))
					if 	diff < avg_diff - 3*std_diff: #and strictly_increasing(last_diffs): # and False:
						pyautogui.rightClick()
						print("tried to capture fish")
						# loot
						self.loot()
						break
				float_prior = float_current
				all_diffs.append(diff)
				c += 1

			print(str(c) + " frames")

			self.jump()
			print("restarting fishing...")

		print('catched ' + str(catched))
		utils.logout()



if __name__ == "__main__":

	bot = WowFishingBot()
	bot.fish()