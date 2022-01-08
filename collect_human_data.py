import cv2
import os

save_base_folder = '/home'

#if not os.path.exists(os.path.join(save_base_folder, 'left')):
#	os.mkdir(os.path.join(save_base_folder, 'left'))
#if not os.path.exists(os.path.join(save_base_folder, 'right')):
#	os.mkdir(os.path.join(save_base_folder, 'right'))

cap = cv2.VideoCapture(0)
left_count=0
right_count=0
while cap.isOpened():
	success, image = cap.read()
	if success:
		cv2.imshow('img', image)
		press_down = cv2.waitKey(1)
		if press_down == ord('l') and left_count < 100:
			cv2.imwrite(os.path.join(save_base_folder, 'left', str(left_count)+'.jpg'), image)
			left_count += 1
		if press_down == ord('r') and right_count < 100:
			cv2.imwrite(os.path.join(save_base_folder, 'right', str(right_count)+'.jpg'), image)
			right_count += 1
		print('left_count is ', left_count)
		print('right count is', right_count)