import cv2
import pyautogui
import numpy as np
import psutil
import win32gui
import win32api, win32con
import time
from skimage.segmentation import slic


def kmeans_apply(image, centroids):
	if len(image.shape) > 2 and image.shape[2] == 4:
		# convert the image from RGBA2RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
	Z = image.reshape((-1, 3))
	for i, point in enumerate(Z):
		Z[i] = centroids[np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])]
	return Z.reshape((image.shape))


def binarize_kmeans(image):
	if len(image.shape) > 2 and image.shape[2] == 4:
		# convert the image from RGBA2RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	z = np.float32(image.reshape((-1, 3)))
	k = 2
	_, label, centroids = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	centroids = np.uint8(centroids)
	res = label.flatten()

	# compute dominant color -> the water
	unique, counts = np.unique(res, return_counts=True)
	water_centroid = np.argmax(counts)

	if water_centroid == 1:
		res = np.logical_not(res).astype(int)

	return res.reshape(image.shape[0:2])


def binarize_red(image):
	if len(image.shape) > 2 and image.shape[2] == 4:
		# convert the image from RGBA2RGB
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

	red_channel = image[:, :, 0]
	green_channel = image[:, :, 1]
	blue_channel = image[:, :, 2]

	binary_red_blue = np.zeros_like(red_channel)
	binary_red_blue[red_channel > blue_channel] = 1

	binary_red_green = np.zeros_like(red_channel)
	binary_red_green[red_channel > green_channel] = 1

	binary_mask = np.multiply(binary_red_blue, binary_red_green)

	return binary_mask  # np.multiply(binary_mask, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def binarize_slic(image):
	labels = slic(image, n_segments=3, compactness=0.1)
	# using 2 segments outputs a blank image, therefore we merge the two dominant clusters as water
	unique, counts = np.unique(labels, return_counts=True)
	bait_cluster = unique[np.argmin(counts)]

	return (labels == bait_cluster).astype('int')


def binarize_canny(image):
	return cv2.Canny(image, 100, 200)


def binarize(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	unique, counts = np.unique(image, return_counts=True)
	water_color = unique[np.argmax(counts)]
	image[image == water_color] = 0
	image[image != 0] = 255

	return image


def strictly_increasing(L):
	return all(x < y for x, y in zip(L, L[1:]))


def distance(p1, p2):
	dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
	return dist


def get_window(window_name):
	def find_wow_window(hwnd, ctx):
		if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == window_name:
			wow_window_handle.append(hwnd)

	wow_window_handle = []
	win32gui.EnumWindows(find_wow_window, None)

	if wow_window_handle:
		return win32gui.GetWindowRect(wow_window_handle[0])
	else:
		return False



def check_process(wow_name):

	program_names = []
	for pid in psutil.pids():
		try:
			program_names.append(psutil.Process(pid).name())
		except:
			pass

	if wow_name in program_names:
		return True
	else:
		return False


def move_mouse(place):
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


def get_subwindow(window, subwindow_coords, mode):

	if mode == "from_corner":
		if all([isinstance(x, float) for x in subwindow_coords]):
			subwindow = window[
						  int(window.shape[0] * subwindow_coords[0]): int(window.shape[0] * subwindow_coords[0]) + int(window.shape[0] * subwindow_coords[2]),
						  int(window.shape[1] * subwindow_coords[1]): int(window.shape[1] * subwindow_coords[1]) + int(window.shape[1] * subwindow_coords[3])]
			return subwindow
		else:
			subwindow = window[
						int(subwindow_coords[0]): int(subwindow_coords[0]) + int(subwindow_coords[2]),
						int(subwindow_coords[1]): int(subwindow_coords[1]) + int(subwindow_coords[3])]
			return subwindow
	elif mode == "pos2neg":
		if all([isinstance(x, float) for x in subwindow_coords]):
			subwindow = window[
						  int(window.shape[0] * subwindow_coords[0]): -int(window.shape[0] * subwindow_coords[2]),
						  int(window.shape[1] * subwindow_coords[1]): -int(window.shape[1] * subwindow_coords[3])]
			return subwindow
		else:
			subwindow = window[
						int(subwindow_coords[0]): -int(subwindow_coords[2]),
						int(subwindow_coords[1]): -int(subwindow_coords[3])]
			return subwindow
	else:
		raise Exception("cannot clip window in mode {}. Use either from_corner or pos2neg".format(mode))


def find_highest_density(points, window_dim):
	scores = []
	for p in points:
		score = 0
		for o in points:
			dist = np.sqrt((p[0] - o[0]) ** 2 + (p[1] - o[1]) ** 2)
			if dist < window_dim:
				score += np.e ** (dist)
		scores.append(score)
	return points[np.argmax(scores)]

