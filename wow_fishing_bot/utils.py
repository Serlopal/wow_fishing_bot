import cv2
import pyautogui
import numpy as np
import psutil
import win32gui

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


