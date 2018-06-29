import logging
import cv2
import pyscreenshot as ImageGrab
import numpy as np
import pyautogui
from matplotlib import pyplot as plt
import numpy as np
import pyaudio
import wave
import audioop
from collections import deque
import os
import time
import psutil
import win32gui
import sys
from sklearn.cluster import KMeans
from mss import mss
from PIL import Image
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import win32api, win32con
import time

dev = False

pyautogui.PAUSE = 0.01

def kmeans_apply(image, centroids):
	if len(image.shape) > 2 and image.shape[2] == 4:
		# convert the image from RGBA2RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
	Z = image.reshape((-1, 3))
	for i,point in enumerate(Z):
		Z[i] = centroids[np.argmin([np.linalg.norm(point-centroid) for centroid in centroids])]
	return Z.reshape((image.shape))

def kmeans_centroids(image):
	if len(image.shape) > 2 and image.shape[2] == 4:
		# convert the image from RGBA2RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	Z = image.reshape((-1, 3))
	# convert to np.float32
	Z = np.float32(Z)
	K = 2
	_, label, centroids = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	centroids = np.uint8(centroids)
	res = centroids[label.flatten()]
	res2 = res.reshape((image.shape))

	return res2

def binarize(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	unique, counts = np.unique(image, return_counts=True)
	water_color = unique[np.argmax(counts)]
	image[image==water_color] = 0
	image[image != 0] = 255

	#plt.imshow(image)
	#plt.show()


	return image
# return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def strictly_increasing(L):
	return all(x<y for x, y in zip(L, L[1:]))

def distance(p1, p2):
	dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
	return dist

def get_window(window_name):
	return win32gui.GetWindowRect(win32gui.FindWindow(None, window_name))

def check_process():
	print('Checking WoW is running')
	if wow_name in [psutil.Process(pid).name() for pid in psutil.pids()]:
		print('WoW is running')
		return True
	else:
		print('WoW is not running')
		return False

def send_float():
	print('Sending float')
	pyautogui.hotkey('8')
	print('Float is sent, waiting animation')
	time.sleep(3)

def jump():
	print('Jump!')
	pyautogui.hotkey(' ')
	time.sleep(1)

def make_screenshot(window):
	print('Capturing screen')
	screenshot = ImageGrab.grab(bbox=window)
	return screenshot

def find_float(img):
	print ('Looking for float')

	#plt.imshow(img)
	#plt.show()

	MIN_MATCH_COUNT = 1

	template = cv2.imread('var/fishing_float_9.png', 0)  # trainImage

	print(template.shape)

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
	good = []
	for m, _ in matches:
		good.append(m)

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

	#plt.imshow(img)
	#plt.scatter(xy[:, 0], xy[:, 1], c='r')
	#a8d8plt.show()

	return np.mean(xy, axis=0)

def move_mouse(place):
	x,y = place[0], place[1]
	print("Moving cursor to " + str(place))
	pyautogui.moveTo(place)

def logout():
	pyautogui.hotkey('return')
	pyautogui.hotkey('1')

	time.sleep(0.1)
	for c in u'/logout':
		time.sleep(0.1)
		pyautogui.hotkey('c')

	time.sleep(0.1)
	pyautogui.hotkey('return')

def clickRight():
	win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
	win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)



if __name__ == "__main__":

	wow_name = 'WoW.exe'
	window_name = 'World of Warcraft'
	# model = load_model('model/model.h5')
	cut_frame = 50 # pixels
	sct = mss()

	loot_coords  = [0.053, 0.30]
	loot_coords_delta = 0.03


	# if check_process() and not dev:
	if True and not dev:
		window = get_window(window_name)
		print("Wow window at " + str(window))
		print("Waiting 2 seconds, so you can switch to WoW")
		time.sleep(2)
		jump()
	else:
		sys.exit()

	for i in range(3):
		time.sleep(1)
		pyautogui.click(x=window[0] + (window[2] - window[0]) * loot_coords[0],
						y=(window[1] + (window[3] - window[1]) * (loot_coords[1] + i * loot_coords_delta)))


	catched = 0
	tries = 0
	c_probs = np.zeros(1400)

	while True:
		tries += 1
		send_float()
		img = np.array(make_screenshot(window).convert('L'))[cut_frame:-cut_frame, cut_frame:-cut_frame]


		# find fishing float in image
		float_coords = find_float(img)

		# if we cannot find the float, try again
		if float_coords is None or np.isnan(float_coords[0]) or np.isnan(float_coords[1]):
			jump()
			continue
		# if we can, let focus on a window around it
		else:
			# coordinates of the float on the image considering WoW window coords and frame removal
			float_coords = [float_coords[0] + window[0] + cut_frame, float_coords[1] + window[1] + cut_frame]
			# move mouse to the float
			move_mouse(float_coords)
			old_diff = 99999
			# counter of sampled images of the float
			c = 0

			# dimension of the window that will encapsule the float
			window_dim = 50
			# capturing of the float window
			window_float = {'top': int(float_coords[1] - window_dim / 2), 'left': int(float_coords[0] - window_dim / 2),
					  'width': window_dim, 'height': window_dim}
			float_prior   = binarize(kmeans_centroids(np.array(sct.grab(window_float))))
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

				# time profiling
				#start = time.time()
				imagen = sct.grab(window_float)
				#a = time.time() - start

				# float_current = binarize(kmeans_apply(np.array(sct.grab(window_float)),centroids))
				float_current = binarize(kmeans_centroids(np.array(sct.grab(window_float))))

				#b = time.time() - start
				#print(b)

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
						for i in range(3):
							time.sleep(1)
							pyautogui.moveTo(x=window[0] + (window[2] - window[0]) * loot_coords[0],
											 y=window[1] + (window[3] - window[1]) * (loot_coords[1] + i * loot_coords_delta),
											 duration=1)
							pyautogui.moveRel(4,0,duration=0.2)
							pyautogui.moveRel(-4,0,duration=0.2)

							pyautogui.click()
						c_probs[c] +=1
						break
				float_prior = float_current
				all_diffs.append(diff)
				c += 1


				# time profiling
				#print(time.time() - start)

			# plt.plot(all_diffs)
			# plt.show()
			print(str(c) + " frames")
			# print(avg_diff)

		jump()
		#save prob vector
		if tries%20==0:
			np.savetxt('prob_vector', c_probs, delimiter=',')
		print("restarting fishing ")

	print('catched ' + str(catched))
	logout()

