import cv2
import numpy as np
import time

# initialize
cap = cv2.VideoCapture(0)
frame_count = 1
name_count = 0
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
actions = [ 'pitch-', 'roll-', 'roll+']

# print the information of the camera
print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

for action in actions:
    while name_count < 99:
        out = cv2.VideoWriter(action +'_'+ str(name_count)+'.avi', fourcc, 30.0, (640, 480))
        while frame_count < 60:
            ret, frame = cap.read()
            frame_copy = cv2.putText(frame, action + str(frame_count)+ '/' + str(60), (20,30), 1, 2, (0,0,255), 3)
            cv2.imshow('frame_color', frame_copy)    # 컬러 화면 출력
            cv2.waitKey(1)
            out.write(frame)
            frame_count += 1

        frame_count = 0
        name_count += 1
        print(name_count)
        out.release()
    name_count = 0
cap.release()
cv2.destroyAllWindows()